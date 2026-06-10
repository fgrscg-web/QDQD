import sys
import os
import io
import traceback
import datetime
import colorsys
import numpy as np
import pandas as pd
import ezdxf
import matplotlib
import matplotlib.pyplot as plt
import pdfplumber
import re

import progress
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams, cm
from matplotlib.ticker import FuncFormatter

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QTextEdit, QFileDialog, QLineEdit,
                               QHBoxLayout, QScrollArea, QFrame, QSplitter, QComboBox,
                               QInputDialog, QMessageBox, QProgressDialog, QRadioButton,QDialog)
from PySide6.QtGui import QTextCursor, QIcon
from PySide6.QtCore import Qt

from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union, polygonize, split, nearest_points, snap
import shapely.affinity as affinity
from shapely.strtree import STRtree
from collections import defaultdict, deque

matplotlib.use('QtAgg')
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False


class SlitViewerDialog(QDialog):
    def __init__(self, main_app, slit_nid):
        super().__init__(main_app)
        self.main_app = main_app
        self.slit_nid = slit_nid
        self.setWindowTitle(f"Slit Node {slit_nid} - Topology Inspector")
        self.resize(700, 600)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            f"<b>Node {slit_nid} (Slit Position)</b><br>"
            f"<span style='color:blue;'>실선(파란색)</span>: 현재 연결이 유지된 경로<br>"
            f"<span style='color:red;'>점선(빨간색)</span>: 루프 탐지로 인해 절단된(Cut) 경로"
        )
        layout.addWidget(info_label)

        # 팝업용 캔버스 생성
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        self.plot_topology()

    def plot_topology(self):
        ax = self.fig.add_subplot(111)
        all_edges = self.main_app.graph_edges

        connected_nodes = set()
        dummy_nodes = set()

        # 1. 파란 실선 (연결 유지) 그리기 및 인접 노드 추적
        for eid, e in self.main_app.remaining_edges_info.items():
            if e['u'] == self.slit_nid or e['v'] == self.slit_nid:
                geom = all_edges[eid]['geometry']
                ax.plot(*geom.xy, color='blue', linewidth=3, alpha=0.8, label='Connected (연결됨)')

                # 원본 슬릿 노드와 이어져 있는 반대쪽 인접 노드 식별
                adj_nid = e['v'] if e['u'] == self.slit_nid else e['u']
                connected_nodes.add(adj_nid)

        # 2. 빨간 점선 (절단/분리) 그리기
        for cut_info in self.main_app.cut_edges_info:
            if cut_info['original_nid'] == self.slit_nid:
                eid = cut_info['edge_id']
                geom = all_edges[eid]['geometry']
                ax.plot(*geom.xy, color='red', linewidth=3, linestyle='--', alpha=0.8, label='Cut (분리됨)')

                # 끊어져 나간 더미 노드 식별
                dummy_nodes.add(cut_info['dummy_nid'])

        # ========================================================
        # 3. 노드 마커 및 정보 라벨 그리기
        # ========================================================

        # A. 메인 슬릿 노드
        center_pt = self.main_app.graph_nodes[self.slit_nid]
        ax.scatter(center_pt[0], center_pt[1], color='lime', marker='*', s=400, edgecolors='black', zorder=10)
        ax.annotate(f"Slit N{self.slit_nid}\n(Main)", (center_pt[0], center_pt[1]),
                    xytext=(12, 12), textcoords='offset points', fontweight='bold', color='green')

        # B. 유지된 인접 노드들
        for nid in connected_nodes:
            if nid == self.slit_nid: continue
            pt = self.main_app.graph_nodes[nid]
            ax.scatter(pt[0], pt[1], color='blue', marker='o', s=100, edgecolors='black', zorder=5)
            ax.annotate(f"N{nid}\n(Adjacent)", (pt[0], pt[1]),
                        xytext=(8, -15), textcoords='offset points', color='blue', fontsize=9)

        # C. 절단되어 새로 생성된 더미 노드들
        for nid in dummy_nodes:
            pt = self.main_app.graph_nodes[nid]
            # 좌표가 원본과 100% 동일하므로 구분을 위해 팝업창에서만 시각적으로 살짝 빗겨나게 표시(Offset)
            offset_x, offset_y = 10.0, 10.0
            ax.scatter(pt[0] + offset_x, pt[1] + offset_y, color='red', marker='X', s=150, edgecolors='black',
                       zorder=10)
            ax.annotate(f"N{nid}\n(Dummy)", (pt[0] + offset_x, pt[1] + offset_y),
                        xytext=(10, 5), textcoords='offset points', color='red', fontweight='bold', fontsize=9)

        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)

        # 범례 중복 방지
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='best')

        self.canvas.draw()


class UltimateShipAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HHI-FAIVE - Floating Automated Integrity Verification and Evaluation")
        self.setWindowIcon(QIcon('icon.ico'))
        self.resize(1800, 1000)
        self.current_dxf_path = ""
        self.is_processing = False
        self.reset_analysis_data()
        self.init_ui()

    def reset_analysis_data(self):
        self.raw_1999_lines = []
        self.left_1999_segments = []
        self.lines_1102 = []
        self.lines_1102_raw = []
        self.lines_157 = []

        # ✨ 새로 추가할 6001~9001 레이어 변수
        self.lines_6001 = []
        self.lines_7001 = []
        self.lines_8001 = []
        self.lines_9001 = []

        self.hull_centroid = Point(0, 0)
        self.is_calculated = False

        self.aligned_internal = []
        self.final_healed_centerlines = []

        self.analysis_nodes = []  # 유효 노드 리스트
        self.analysis_elements = []  # 유효 1D 요소(부재) 리스트

        self.graph_nodes = {}
        self.graph_edges = []
        self.graph_summary = ""

        self.flowchart_slit_nodes = []

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Malgun Gothic', 'Noto Sans KR', sans-serif;
                font-size: 12px;
                color: #2C3E50;
            }
            QLineEdit, QComboBox {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #CED4DA;
                border-radius: 4px;
                padding: 2px 5px;
                min-height: 22px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1.5px solid #00AD1D;
            }
            QFrame#settingBox {
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 6px;
                padding: 5px;
                margin-top: 5px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
            QPushButton {
                font-family: 'Malgun Gothic';
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
                padding: 2px;
                margin-top: 5px;
                margin-bottom: 5px;
            }
            QPushButton#btnGreen {
                background-color: #00AD1D;
                color: white;
            }
            QPushButton#btnGreen:hover {
                background-color: #009619;
            }
            QPushButton#btnGreen:disabled {
                background-color: #A5D6A7;
                color: #F1F8E9;
                border: none;
            }
            QTextEdit#resultBox {
                font-family: 'Consolas', monospace;
                font-size: 13px;
                background-color: #FDFEFE;
                border: 1px solid #CED4DA;
                border-radius: 2px;
                padding: 5px;
            }
        """)

        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)

        self.field_width = 100

        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_panel_layout = QVBoxLayout(control_panel)
        control_panel_layout.setAlignment(Qt.AlignTop)

        # [General Settings]
        gen_box = QFrame()
        gen_box.setObjectName("settingBox")
        gen_vbox = QVBoxLayout(gen_box)
        gen_vbox.addWidget(QLabel("<b>[General Settings]</b>"))

        # ✨ "Total Vy (t):" 로 단위 변경, 기본값 100(톤)으로 설정
        for lbl, attr, dval in [("Scale:", "txt_scale", "100"), ("H-Ext (mm):", "txt_ext", "10"),
                                ("V-Ext (mm):", "txt_perp", "10"), ("Total Vy (t):", "txt_vy", "100")]:
            h = QHBoxLayout()
            h.addWidget(QLabel(lbl))
            h.addStretch()
            le = QLineEdit(dval)
            le.setFixedWidth(self.field_width)
            setattr(self, attr, le)
            h.addWidget(le)
            gen_vbox.addLayout(h)
        control_panel_layout.addWidget(gen_box)

        self.btn_load = QPushButton("1. DXF Load 📂")
        self.btn_load.setFixedHeight(40)
        self.btn_load.setObjectName("btnGreen")
        self.btn_load.clicked.connect(self.load_and_process_dxf)
        control_panel_layout.addWidget(self.btn_load)

        self.btn_calc = QPushButton("2. 1D 변환 및 정렬 📐")
        self.btn_calc.setFixedHeight(40)
        self.btn_calc.setObjectName("btnGreen")
        self.btn_calc.clicked.connect(self.process_1d_geometry)
        control_panel_layout.addWidget(self.btn_calc)

        control_panel_layout.addStretch()

        main_layout.addWidget(control_panel)

        # ---------------------------------------------------------
        # 작업 영역 (렌더링 및 출력)
        # ---------------------------------------------------------
        work_area = QWidget()
        work_layout = QVBoxLayout(work_area)
        viz_splitter = QSplitter(Qt.Horizontal)

        # 뷰포트 타이틀 변경
        for i, title in enumerate(["[Final Healed 1D Geometry]", "[Graphified Hull Cross-Section]"]):
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.addWidget(QLabel(f"<b>{title}</b>"))
            fig = Figure()
            can = FigureCanvas(fig)
            lay.addWidget(NavigationToolbar(can, self))
            lay.addWidget(can, stretch=1)
            setattr(self, f"fig{i + 1}", fig)
            setattr(self, f"can{i + 1}", can)
            viz_splitter.addWidget(container)

            # ✨ 두 번째 캔버스(그래프 뷰)에 클릭 이벤트 핸들러 연결
            if i == 1:
                can.mpl_connect('pick_event', self.on_edge_click)
                can.mpl_connect('pick_event', self.on_slit_node_click)

        work_layout.addWidget(viz_splitter, stretch=7)

        self.result_box = QTextEdit()
        self.result_box.setObjectName("resultBox")
        self.result_box.setReadOnly(True)
        self.result_box.setFixedHeight(200)
        work_layout.addWidget(self.result_box)
        main_layout.addWidget(work_area, stretch=7)

        main_scroll.setWidget(main_container)
        self.setCentralWidget(main_scroll)

    # =====================================================================
    # 부재 판단 및 그룹/정렬 메서드
    # =====================================================================
    def group_and_align_centerlines(self, centerlines, tol_dist=150.0, tol_angle=1.5):
        v_lines, h_lines, d_lines = [], [], []

        # 1. 수평(H), 수직(V), 대각선(D) 엄격한 분류
        for cl in centerlines:
            coords = list(cl['line'].coords)
            max_seg_len = 0
            best_angle = 0
            total_len = 0

            for i in range(len(coords) - 1):
                p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
                v = p2 - p1
                length = np.linalg.norm(v)
                total_len += length

                if length > max_seg_len:
                    max_seg_len = length
                    best_angle = np.degrees(np.arctan2(v[1], v[0])) % 180

            if total_len < 1e-6: continue
            cl_info = {'cl': cl, 'length': total_len, 'coords': coords, 'angle': best_angle}

            if best_angle <= tol_angle or best_angle >= 180 - tol_angle:
                h_lines.append(cl_info)
            elif 90 - tol_angle <= best_angle <= 90 + tol_angle:
                v_lines.append(cl_info)
            else:
                d_lines.append(cl_info)

        import colorsys
        aligned_centerlines = []

        # 2. 수직 부재(V) 그룹화 및 정렬
        v_groups = []
        for info in v_lines:
            mid_x = (info['coords'][0][0] + info['coords'][-1][0]) / 2.0
            placed = False
            for g in v_groups:
                if abs(g['avg_x'] - mid_x) < tol_dist:
                    g['members'].append(info)
                    tot_len = sum(m['length'] for m in g['members'])
                    g['avg_x'] = sum(
                        ((m['coords'][0][0] + m['coords'][-1][0]) / 2.0) * m['length'] for m in g['members']) / tot_len
                    placed = True
                    break
            if not placed:
                v_groups.append(
                    {'avg_x': mid_x, 'members': [info], 'color': colorsys.hsv_to_rgb(np.random.rand(), 0.8, 0.9)})

        for g in v_groups:
            avg_x = g['avg_x']
            for m in g['members']:
                new_coords = [(avg_x, p[1]) for p in m['coords']]
                m['cl']['line'] = LineString(new_coords)
                m['cl']['color'] = g['color']
                aligned_centerlines.append(m['cl'])

        # 3. 수평 부재(H) 그룹화 및 정렬
        h_groups = []
        for info in h_lines:
            mid_y = (info['coords'][0][1] + info['coords'][-1][1]) / 2.0
            placed = False
            for g in h_groups:
                if abs(g['avg_y'] - mid_y) < tol_dist:
                    g['members'].append(info)
                    tot_len = sum(m['length'] for m in g['members'])
                    g['avg_y'] = sum(
                        ((m['coords'][0][1] + m['coords'][-1][1]) / 2.0) * m['length'] for m in g['members']) / tot_len
                    placed = True
                    break
            if not placed:
                h_groups.append(
                    {'avg_y': mid_y, 'members': [info], 'color': colorsys.hsv_to_rgb(np.random.rand(), 0.8, 0.9)})

        for g in h_groups:
            avg_y = g['avg_y']
            for m in g['members']:
                new_coords = [(p[0], avg_y) for p in m['coords']]
                m['cl']['line'] = LineString(new_coords)
                m['cl']['color'] = g['color']
                aligned_centerlines.append(m['cl'])

        # ✨ 4. 대각선 부재(D) 그룹화 (각도와 원점 거리를 이용한 직선 방정식 기반)
        d_groups = []
        for info in d_lines:
            p1 = np.array(info['coords'][0])
            ang = info['angle']
            th = np.radians(ang)
            # 원점에서 직선까지의 수직 거리 (rho)
            rho = -p1[0] * np.sin(th) + p1[1] * np.cos(th)

            placed = False
            for g in d_groups:
                # 각도 오차 및 수직 거리(평행 간격) 오차 확인
                ang_diff = min(abs(g['avg_angle'] - ang), 180 - abs(g['avg_angle'] - ang))
                if ang_diff < tol_angle and abs(g['avg_rho'] - rho) < tol_dist:
                    g['members'].append(info)
                    tot_len = sum(m['length'] for m in g['members'])
                    # 각도 가중 평균 갱신
                    g['avg_angle'] = sum(m['angle'] * m['length'] for m in g['members']) / tot_len

                    # 갱신된 각도로 그룹의 평균 rho 재계산
                    avg_th = np.radians(g['avg_angle'])
                    new_rho_sum = 0
                    for m in g['members']:
                        mp1 = np.array(m['coords'][0])
                        m_rho = -mp1[0] * np.sin(avg_th) + mp1[1] * np.cos(avg_th)
                        new_rho_sum += m_rho * m['length']
                    g['avg_rho'] = new_rho_sum / tot_len

                    placed = True
                    break

            if not placed:
                d_groups.append({
                    'avg_angle': ang,
                    'avg_rho': rho,
                    'members': [info],
                    'color': colorsys.hsv_to_rgb(np.random.rand(), 0.8, 0.9)  # 그룹별 색상 부여
                })

        # ✨ 5. 대각선 부재 정렬 (계산된 평균 직선 위로 좌표 투영/Projection)
        for g in d_groups:
            th = np.radians(g['avg_angle'])
            rho = g['avg_rho']

            # 직선의 방향 벡터와 기준점
            dir_v = np.array([np.cos(th), np.sin(th)])
            p0 = np.array([-rho * np.sin(th), rho * np.cos(th)])

            for m in g['members']:
                new_coords = []
                for p in m['coords']:
                    pt = np.array(p)
                    # 현재 점을 평균 직선 위로 직교 투영(Orthogonal Projection)
                    t = np.dot(pt - p0, dir_v)
                    proj_pt = p0 + t * dir_v
                    new_coords.append(tuple(proj_pt))

                m['cl']['line'] = LineString(new_coords)
                m['cl']['color'] = g['color']
                aligned_centerlines.append(m['cl'])

        return aligned_centerlines

    # =====================================================================
    # 유틸리티 메서드
    # =====================================================================
    def _extract_pts(self, e, scale):
        try:
            if e.dxftype() == 'LINE':
                return [(e.dxf.start.x * scale, e.dxf.start.y * scale), (e.dxf.end.x * scale, e.dxf.end.y * scale)]
            elif e.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                return [(p[0] * scale, p[1] * scale) for p in e.get_points()]
        except:
            return None
        return None

    def heal_internal_collinear(self, lines, threshold_gap=150.0):
        """내부 부재(1102, 157 등)에 대해 150mm 이내의 동일 선상 틈새를 이어주는 함수"""
        if not lines: return []
        bridges = []
        groups = {}
        for l in lines:
            c = list(l.coords)
            p1, p2 = np.array(c[0]), np.array(c[-1])
            v = p2 - p1
            L = np.linalg.norm(v)
            if L < 1e-6: continue

            # 각도와 거리(rho)를 기준으로 동일 선상(Collinear) 판별
            a = np.degrees(np.arctan2(v[1], v[0])) % 180.0
            ak = round(a, 0)
            th = np.radians(ak)
            rho = round((-p1[0] * np.sin(th) + p1[1] * np.cos(th)) / 10.0) * 10.0
            key = (ak, rho)

            if key not in groups: groups[key] = []
            groups[key].append((l, p1, p2))

        for (ak, _), grp in groups.items():
            if len(grp) < 2: continue
            dv = np.array([np.cos(np.radians(ak)), np.sin(np.radians(ak))])
            # 선분의 진행 방향으로 정렬
            segs = sorted([(np.dot(p1, dv), np.dot(p2, dv), p1, p2) for _, p1, p2 in grp],
                          key=lambda x: min(x[0], x[1]))

            # 정렬된 선분들 사이의 틈새 계산
            for i in range(len(segs) - 1):
                pe = segs[i][3] if segs[i][0] > segs[i][1] else segs[i][2]
                pn = segs[i + 1][2] if segs[i + 1][0] > segs[i + 1][1] else segs[i + 1][3]
                g = np.linalg.norm(pn - pe)

                # 틈새가 임계값(150mm) 이내인 경우 이어줌
                if 0.1 < g <= threshold_gap:
                    bridges.append(LineString([tuple(pe), tuple(pn)]))
        return lines + bridges

    def robust_heal_1999(self, line_infos, max_gap=500.0):
        """
        [완전판] 외판(1999) 조각들을 수학적으로 결합한 뒤,
        닫히지 않은 틈새(자신의 꼬리를 무는 틈새 포함)를 찾아 무조건 닫아줍니다.
        """
        if not line_infos: return []

        from shapely.ops import linemerge
        import numpy as np
        from shapely.geometry import LineString

        # 1. 쪼개진 1999 조각들을 일단 수학적으로 결합(Merge)하여 최대한 덩어리를 키웁니다.
        raw_lines = [info['line'] for info in line_infos]
        merged_geom = linemerge(raw_lines)

        if merged_geom.geom_type == 'LineString':
            merged_lines = [merged_geom]
        elif merged_geom.geom_type == 'MultiLineString':
            merged_lines = list(merged_geom.geoms)
        else:
            merged_lines = raw_lines

        new_infos = []
        for geom in merged_lines:
            new_infos.append({
                'line': geom,
                'thickness': 10.0,
                'type': '1999',
                'color': '#333333'
            })

        # 2. "진짜 끝점"들만 추출 (폐곡선으로 닫히지 않은 부분)
        endpoints = []
        for i, info in enumerate(new_infos):
            c = list(info['line'].coords)
            if len(c) < 2: continue
            # 이미 닫힌 루프(원)가 아니라면 양 끝점을 틈새 후보로 등록
            if np.linalg.norm(np.array(c[0]) - np.array(c[-1])) > 1e-3:
                endpoints.append({'idx': i, 'pt': np.array(c[0])})
                endpoints.append({'idx': i, 'pt': np.array(c[-1])})

        used_pts = set()
        bridges = []

        # 3. 끝점들끼리 최단 거리 짝 찾기 (💡자신의 시작점과 끝점을 연결하는 것도 허용!)
        for i, ep1 in enumerate(endpoints):
            if i in used_pts: continue

            best_j = -1
            best_dist = max_gap  # 기본 500mm 틈새까지 모두 추적

            for j, ep2 in enumerate(endpoints):
                if i == j or j in used_pts: continue

                dist = np.linalg.norm(ep1['pt'] - ep2['pt'])
                if dist <= best_dist:
                    best_dist = dist
                    best_j = j

            # 4. 짝을 찾았다면 붉은색 브릿지로 강제 연결
            if best_j != -1:
                ep2 = endpoints[best_j]
                bridges.append({
                    'line': LineString([tuple(ep1['pt']), tuple(ep2['pt'])]),
                    'thickness': 10.0,
                    'type': '1999',
                    # ✨ 어디가 힐링되었는지 눈으로 명확히 보이도록 연결부만 [빨간색]으로 칠합니다!
                    'color': '#FF0000'
                })
                used_pts.add(i)
                used_pts.add(best_j)

        return new_infos + bridges

    # =====================================================================
    # 메인 프로세스
    # =====================================================================
    def load_and_process_dxf(self):
        if self.is_processing: return
        fname, _ = QFileDialog.getOpenFileName(self, 'Select DXF File', '', 'DXF files (*.dxf)')
        if not fname: return
        fname = os.path.abspath(os.path.normpath(fname))
        self.reset_analysis_data()
        self.result_box.clear()
        self.current_dxf_path = fname
        try:
            scale = float(self.txt_scale.text())
            try:
                doc = ezdxf.readfile(fname, encoding='cp949')
            except:
                try:
                    doc = ezdxf.readfile(fname, encoding='utf-8')
                except:
                    doc = ezdxf.readfile(fname)

            msp = doc.modelspace()
            active_layers = {l.dxf.name for l in doc.layers if l.is_on() and not l.is_frozen()}
            t_1999, t_1204 = [], []
            t_layers = {"-1102": [], "157": [], "6001": [], "7001": [], "8001": [], "9001": []}

            for e in msp:
                layer = e.dxf.layer.strip()
                if layer not in active_layers: continue
                pts = self._extract_pts(e, scale)
                if not pts or len(pts) < 2: continue
                ls = LineString(pts)
                if layer == "1999":
                    t_1999.append(ls)
                elif layer == "-1204":
                    t_1204.append(ls)
                elif layer in t_layers:
                    t_layers[layer].append(ls)

            if t_1999:
                u_1999 = unary_union(t_1999)
                self.cx = u_1999.centroid.x
                self.cy_base = u_1999.bounds[1]
            else:
                self.cx = self.cy_base = 0.0

            shift = lambda ls: LineString([(p[0] - self.cx, p[1] - self.cy_base) for p in ls.coords])

            # (기존) 1999 외판 단일화
            self.raw_1999_lines = [shift(ls) for ls in t_1999]
            m_1999 = unary_union(self.raw_1999_lines)
            self.hull_centroid = m_1999.centroid

            # ✨ 수정: 커터(Cutter) 강화 - 1204 레이어와 x=0을 이용한 완벽한 분할
            cutters = []

            # 1. -1204 레이어를 양방향으로 1000mm씩 과연장하여 외판을 확실히 절단
            for c in [shift(ls) for ls in t_1204]:
                c_coords = list(c.coords)
                p1, p2 = np.array(c_coords[0]), np.array(c_coords[-1])
                v = p2 - p1
                L = np.linalg.norm(v)
                if L > 1e-6:
                    u = v / L
                    # 확실한 교차를 위해 선분을 양쪽으로 길게 늘림
                    cutters.append(LineString([tuple(p1 - u * 1000), tuple(p2 + u * 1000)]))

            # 2. x=0 (Centerline) 커터 추가 (상하로 2000mm 연장)
            y_min, y_max = m_1999.bounds[1], m_1999.bounds[3]
            center_cutter = LineString([(0, y_min - 2000), (0, y_max + 2000)])
            cutters.append(center_cutter)

            # 3. 분할 실행
            split_res = split(m_1999, unary_union(cutters)) if cutters else m_1999
            pieces = list(split_res.geoms) if hasattr(split_res, 'geoms') else [split_res]

            # 4. 파편 정리 (50mm 이하의 쓸모없는 찌꺼기는 버리고 의미 있는 외판 조각만 취합)
            self.left_1999_segments = []
            for g in pieces:
                if g.length > 50.0:
                    self.left_1999_segments.append(g)

            self.left_1999_segments.sort(key=lambda s: (-round(s.centroid.y, 2), s.centroid.x))

            self.lines_1102 = [shift(ls) for ls in t_layers["-1102"]]
            self.lines_1102_raw = list(self.lines_1102)
            self.lines_157 = [shift(ls) for ls in t_layers["157"]]

            self.lines_6001 = [shift(ls) for ls in t_layers["6001"]]
            self.lines_7001 = [shift(ls) for ls in t_layers["7001"]]
            self.lines_8001 = [shift(ls) for ls in t_layers["8001"]]
            self.lines_9001 = [shift(ls) for ls in t_layers["9001"]]
            self.refresh_ui()
            self.result_box.append(f"✅ Successfully loaded: {os.path.basename(fname)}")
        except Exception as e:
            self.result_box.setText(f"❌ Load Error Detailed:\n{traceback.format_exc()}")

    def process_1d_geometry(self):
        if self.is_processing: return
        self.is_processing = True
        self.btn_calc.setEnabled(False)
        self.btn_load.setEnabled(False)

        progress = QProgressDialog("1D Transformation Processing...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing...")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()
        QApplication.processEvents()

        # 도우미 함수 ---------------------------------------------------
        def filter_short(lines, ml=100.0):
            return [l for l in lines if l.length >= ml]

        def remove_overlapping(lines, dt=10.0, at=5.0):
            lines = sorted(lines, key=lambda x: x.length, reverse=True)
            kept = []
            kept_meta = []
            for i, l in enumerate(lines):
                if i % 10 == 0: QApplication.processEvents()
                c = list(l.coords)
                ps, pe = np.array(c[0]), np.array(c[-1])
                v = pe - ps
                ln = np.linalg.norm(v)
                if ln < 1e-6: continue
                ang = np.degrees(np.arctan2(v[1], v[0])) % 180
                dup = False
                for km in kept_meta:
                    ak, pk1, vk, lk = km['ang'], km['ps'], km['v'], km['ln']
                    if min(abs(ang - ak), 180 - abs(ang - ak)) > at: continue
                    vu = vk / lk
                    mid = (ps + pe) / 2.0
                    if np.linalg.norm(mid - (pk1 + np.dot(mid - pk1, vu) * vu)) > dt: continue
                    t1, t2 = np.dot(ps - pk1, vu), np.dot(pe - pk1, vu)
                    if min(lk, max(t1, t2)) - max(0, min(t1, t2)) > ln * 0.8:
                        dup = True
                        break
                if not dup:
                    kept.append(l)
                    kept_meta.append({'ang': ang, 'ps': ps, 'v': v, 'ln': ln})
            return kept

        def split_by_slope(line, at=5.0):
            coords = list(line.coords)
            if len(coords) < 3: return [line]
            segs = []
            cur = [coords[0]]
            for i in range(1, len(coords) - 1):
                cur.append(coords[i])
                v1 = np.array(coords[i]) - np.array(coords[i - 1])
                v2 = np.array(coords[i + 1]) - np.array(coords[i])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6: continue
                a = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
                if a > at:
                    segs.append(LineString(cur))
                    cur = [coords[i]]
            cur.append(coords[-1])
            if len(cur) >= 2: segs.append(LineString(cur))
            return segs

        def match_pairs(lines, max_dist=100.0, angle_tol=20.0, overlap_tolerance=5.0):
            if not lines: return []
            ls_sorted = sorted(lines, key=lambda x: x.length)
            meta = []
            for i, l in enumerate(ls_sorted):
                if i % 10 == 0: QApplication.processEvents()
                c = list(l.coords)
                ps, pe = np.array(c[0]), np.array(c[-1])
                v = pe - ps
                ln = np.linalg.norm(v)
                if ln < 1e-6:
                    meta.append(None)
                    continue
                minx, miny = min(ps[0], pe[0]), min(ps[1], pe[1])
                maxx, maxy = max(ps[0], pe[0]), max(ps[1], pe[1])
                meta.append({'ps': ps, 'pe': pe, 'v': v, 'ln': ln,
                             'unit': v / ln, 'ang': np.degrees(np.arctan2(v[1], v[0])) % 180,
                             'mid': (ps + pe) / 2.0,
                             'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy})
            used = {i: [] for i in range(len(ls_sorted))}
            pairs = []
            for i in range(len(ls_sorted)):
                if i % 10 == 0:
                    QApplication.processEvents()
                    if progress.wasCanceled(): raise UserWarning("User canceled.")
                if meta[i] is None: continue
                mi = meta[i]
                best_j, best_d, best_ov = -1, float('inf'), None
                mi_expand_minx = mi['minx'] - max_dist
                mi_expand_maxx = mi['maxx'] + max_dist
                mi_expand_miny = mi['miny'] - max_dist
                mi_expand_maxy = mi['maxy'] + max_dist

                for j in range(i + 1, len(ls_sorted)):
                    if meta[j] is None: continue
                    mj = meta[j]
                    if (mj['maxx'] < mi_expand_minx or mj['minx'] > mi_expand_maxx or
                            mj['maxy'] < mi_expand_miny or mj['miny'] > mi_expand_maxy):
                        continue
                    ad = min(abs(mi['ang'] - mj['ang']), 180 - abs(mi['ang'] - mj['ang']))
                    if ad > angle_tol: continue
                    proj_infinite = mj['ps'] + np.dot(mi['mid'] - mj['ps'], mj['unit']) * mj['unit']
                    d = np.linalg.norm(mi['mid'] - proj_infinite)
                    if d > max_dist: continue
                    t1 = np.dot(mi['ps'] - mj['ps'], mj['unit'])
                    t2 = np.dot(mi['pe'] - mj['ps'], mj['unit'])
                    ov_s, ov_e = max(0, min(t1, t2)), min(mj['ln'], max(t1, t2))
                    if (ov_e - ov_s) < mi['ln'] * 0.1: continue
                    is_blocked = False
                    for (us, ue) in used[j]:
                        if min(ov_e, ue) - max(ov_s, us) > overlap_tolerance:
                            is_blocked = True
                            break
                    if is_blocked: continue
                    if d < best_d:
                        best_d, best_j, best_ov = d, j, (ov_s, ov_e)
                if best_j >= 0 and best_ov:
                    used[best_j].append(best_ov)
                    pairs.append((i, best_j, best_ov, best_d))
            return [(ls_sorted[i], ls_sorted[j], ov, dist) for i, j, ov, dist in pairs]

        def create_centerlines(pairs):
            result = []
            for i, (short_line, long_line, (ov_s, ov_e), dist) in enumerate(pairs):
                if i % 10 == 0: QApplication.processEvents()
                cs = list(short_line.coords)
                cl_c = list(long_line.coords)
                ps1, ps2 = np.array(cs[0]), np.array(cs[-1])
                pl1 = np.array(cl_c[0])
                vl = np.array(cl_c[-1]) - pl1
                ll = np.linalg.norm(vl)
                if ll < 1e-6: continue
                vl_u = vl / ll
                mids = []
                for frac in np.linspace(0, 1, 5):
                    pt_s = ps1 + (ps2 - ps1) * frac
                    t = np.dot(pt_s - pl1, vl_u)
                    pt_l = pl1 + t * vl_u
                    mids.append(tuple((pt_s + pt_l) / 2.0))
                result.append({'line': LineString(mids), 'thickness': round(dist * 2) / 2.0})
            return result

        def create_continuous_stiffener_centerlines(pairs):
            long_line_map = {}
            for short_line, long_line, (ov_s, ov_e), dist in pairs:
                idx = id(long_line)
                if idx not in long_line_map:
                    long_line_map[idx] = {'long_line': long_line, 'shorts': [], 'dist': dist}
                long_line_map[idx]['shorts'].append(short_line)

            result = []
            for data in long_line_map.values():
                ll = data['long_line']
                dist = data['dist']
                shorts = data['shorts']
                cl_c = list(ll.coords)
                p1, p2 = np.array(cl_c[0]), np.array(cl_c[-1])
                v = p2 - p1
                length = np.linalg.norm(v)
                if length < 1e-6: continue
                vu = v / length

                sl = shorts[0]
                sc = list(sl.coords)
                ps_mid = (np.array(sc[0]) + np.array(sc[-1])) / 2.0
                pl_mid = (p1 + p2) / 2.0

                vec = ps_mid - pl_mid
                proj = np.dot(vec, vu) * vu
                perp = vec - proj
                perp_len = np.linalg.norm(perp)

                if perp_len < 1e-6:
                    n = np.array([-vu[1], vu[0]])
                else:
                    n = perp / perp_len

                offset = n * (dist / 2.0)
                new_coords = [tuple(np.array(pt) + offset) for pt in cl_c]
                result.append({'line': LineString(new_coords), 'thickness': round(dist * 2) / 2.0})
            return result

        # 1차 보정: 10mm 제한 연장 및 삐져나감 정리 알고리즘
        def extend_and_trim_10mm(centerlines):
            base_geoms = [cl['line'] for cl in centerlines]
            open_ends = []

            for i, cl in enumerate(centerlines):
                coords = list(cl['line'].coords)
                for ei in [0, -1]:
                    p = np.array(coords[ei])
                    nb = 1 if ei == 0 else -2
                    v = p - np.array(coords[nb])
                    ln = np.linalg.norm(v)
                    if ln < 1e-6: continue
                    d = v / ln

                    pt = Point(p)
                    is_open = True
                    for j, bg in enumerate(base_geoms):
                        if i == j: continue
                        if bg.distance(pt) < 1e-3:
                            is_open = False
                            break

                    if is_open:
                        ray = LineString([tuple(p), tuple(p + d * 10.0)])
                        open_ends.append({
                            'line_idx': i, 'end_idx': ei, 'p': p, 'd': d, 'ray': ray
                        })

            updates = {}
            for k, oe in enumerate(open_ends):
                ray = oe['ray']
                p = oe['p']
                best_dist = 10.0 + 1e-5
                best_p = None

                for j, bg in enumerate(base_geoms):
                    if oe['line_idx'] == j: continue
                    if ray.intersects(bg):
                        inter = ray.intersection(bg)
                        pts = [inter] if inter.geom_type == 'Point' else list(
                            inter.geoms) if inter.geom_type == 'MultiPoint' else []
                        for pt_int in pts:
                            dist = np.linalg.norm(np.array([pt_int.x, pt_int.y]) - p)
                            if 1e-3 < dist < best_dist:
                                best_dist = dist
                                best_p = (pt_int.x, pt_int.y)

                for j, other_oe in enumerate(open_ends):
                    if k == j: continue
                    if oe['line_idx'] == other_oe['line_idx']: continue
                    other_ray = other_oe['ray']
                    if ray.intersects(other_ray):
                        inter = ray.intersection(other_ray)
                        pts = [inter] if inter.geom_type == 'Point' else list(
                            inter.geoms) if inter.geom_type == 'MultiPoint' else []
                        for pt_int in pts:
                            dist = np.linalg.norm(np.array([pt_int.x, pt_int.y]) - p)
                            if 1e-3 < dist < best_dist:
                                best_dist = dist
                                best_p = (pt_int.x, pt_int.y)

                if best_p is not None:
                    if oe['line_idx'] not in updates:
                        updates[oe['line_idx']] = {}
                    updates[oe['line_idx']][oe['end_idx']] = best_p

            new_centerlines = []
            for i, cl in enumerate(centerlines):
                if i in updates:
                    coords = list(cl['line'].coords)
                    if 0 in updates[i]: coords[0] = updates[i][0]
                    if -1 in updates[i]: coords[-1] = updates[i][-1]
                    new_cl = cl.copy()
                    new_cl['line'] = LineString(coords)
                    new_centerlines.append(new_cl)
                else:
                    new_centerlines.append(cl)
            return new_centerlines

        # ✨ [부활] 2차 보정: 레이 캐스팅 (10mm 이상의 먼 거리 연장)
        def raycast_extend(centerlines, max_dist=300.0):
            extended_pts = []
            result = []

            # ✨ 1. 동적 충돌 맵(Dynamic Collision Map) 생성
            # 이전 부재가 연장된 결과를 다음 부재가 즉시 인식할 수 있도록 복사본 생성
            current_geoms = [cl['line'] for cl in centerlines]
            bounds = [g.bounds for g in current_geoms]

            for i, cl in enumerate(centerlines):
                if i % 10 == 0: QApplication.processEvents()
                coords = list(cl['line'].coords)
                if len(coords) < 2:
                    result.append(cl)
                    continue

                for ei in [0, -1]:
                    p = np.array(coords[ei])
                    nb = 1 if ei == 0 else -2
                    v = p - np.array(coords[nb])
                    vn = np.linalg.norm(v)
                    if vn < 1e-6: continue
                    d = v / vn

                    # ✨ 2. 데드존 해결: 실제 선분(LineString)과의 최단 거리가 1.0mm(병합 허용치) 이내인지 검사
                    pt_geom = Point(p)
                    conn = False
                    for j, o_geom in enumerate(current_geoms):
                        if i == j: continue
                        # 끝점뿐만 아니라 선분 중간에 닿아 있어도 완벽히 인식
                        if pt_geom.distance(o_geom) <= 1.0:
                            conn = True
                            break
                    if conn: continue

                    # ✨ 3. 원본(centerlines)이 아닌 동적 맵(current_geoms)을 타겟으로 Ray 발사
                    ray = LineString([tuple(p), tuple(p + d * max_dist)])
                    rb = ray.bounds
                    bp, bd = None, max_dist

                    for j, o_geom in enumerate(current_geoms):
                        if i == j: continue
                        ob = bounds[j]
                        if rb[2] < ob[0] or rb[0] > ob[2] or rb[3] < ob[1] or rb[1] > ob[3]:
                            continue

                        inter = ray.intersection(o_geom)
                        if inter.is_empty: continue

                        pts = []
                        if inter.geom_type == 'Point':
                            pts = [inter]
                        elif inter.geom_type == 'MultiPoint':
                            pts = list(inter.geoms)
                        elif inter.geom_type == 'LineString':
                            pts = [Point(inter.coords[0]), Point(inter.coords[-1])]

                        for pt in pts:
                            dd = np.linalg.norm(np.array([pt.x, pt.y]) - p)
                            if 1e-3 < dd < bd:
                                bd = dd
                                bp = (pt.x, pt.y)

                    if bp:
                        overshoot = 0.1  # 확실한 교차 분할을 위한 과연장
                        bp_o = (bp[0] + d[0] * overshoot, bp[1] + d[1] * overshoot)
                        if ei == 0:
                            coords[0] = bp_o
                        else:
                            coords[-1] = bp_o
                        extended_pts.append(np.array(bp_o))

                # ✨ 4. 현재 선분이 연장되었다면, 그 결과를 즉시 동적 맵(current_geoms)에 업데이트!
                new_line = LineString(coords)
                current_geoms[i] = new_line
                bounds[i] = new_line.bounds

                new_cl = cl.copy()
                new_cl['line'] = new_line
                result.append(new_cl)

            return result, extended_pts

        def split_all_lines_at_intersections(centerlines):
            all_line_geoms = [cl['line'] for cl in centerlines]
            bounds = [g.bounds for g in all_line_geoms]
            intersection_points = []
            for i in range(len(all_line_geoms)):
                b1 = bounds[i]
                for j in range(i + 1, len(all_line_geoms)):
                    b2 = bounds[j]
                    if b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]: continue
                    try:
                        inter = all_line_geoms[i].intersection(all_line_geoms[j])
                        if inter.is_empty: continue
                        if inter.geom_type == 'Point':
                            intersection_points.append(inter)
                        elif inter.geom_type == 'MultiPoint':
                            intersection_points.extend(inter.geoms)
                        # ✨ [핵심 추가] 동일 선상에서 겹쳐 선(LineString)으로 교차할 때 끝점 추출
                        elif inter.geom_type == 'LineString':
                            intersection_points.append(Point(inter.coords[0]))
                            intersection_points.append(Point(inter.coords[-1]))
                        elif inter.geom_type == 'MultiLineString':
                            for ls in inter.geoms:
                                intersection_points.append(Point(ls.coords[0]))
                                intersection_points.append(Point(ls.coords[-1]))
                    except:
                        pass

            for g in all_line_geoms:
                intersection_points.append(Point(g.coords[0]))
                intersection_points.append(Point(g.coords[-1]))

            if not intersection_points: return centerlines

            unique_points = []
            for pt in intersection_points:
                if not unique_points or min(pt.distance(upt) for upt in unique_points) > 1e-3:
                    unique_points.append(pt)
            splitter = unary_union(unique_points)

            new_centerlines = []
            for cl in centerlines:
                line = cl['line']
                try:
                    snapped_line = snap(line, splitter, 0.05)
                    res = split(snapped_line, splitter)
                    geoms = list(res.geoms) if hasattr(res, 'geoms') else [res]
                    for geom in geoms:
                        new_cl = cl.copy()
                        new_cl['line'] = geom
                        new_centerlines.append(new_cl)
                except:
                    new_centerlines.append(cl)
            return new_centerlines

        def clean_topology(centerlines, trim_tol=15.0):
            # 1. 중복 선분 제거 (이중 두께 및 중복 계산 방지)
            unique_cl = []
            seen = set()

            for cl in centerlines:
                c = list(cl['line'].coords)
                if len(c) < 2: continue

                p1, p2 = c[0], c[-1]
                # ✨ 소수점 정밀도를 높여 멀쩡한 노드가 서로 다른 점으로 인식되는 현상 방지
                p1_r = (round(p1[0], 3), round(p1[1], 3))
                p2_r = (round(p2[0], 3), round(p2[1], 3))

                # 0.5 미만의 미세한 찌꺼기 선분만 무시
                if np.hypot(p2_r[0] - p1_r[0], p2_r[1] - p1_r[1]) < 0.5:
                    continue

                seg_key = tuple(sorted([p1_r, p2_r]))

                if seg_key not in seen:
                    seen.add(seg_key)
                    unique_cl.append(cl)

            # 2. 삐져나온 꼬리(Dangling) 반복 트림
            while True:
                endpoints = []
                for cl in unique_cl:
                    c = list(cl['line'].coords)
                    endpoints.extend([(round(c[0][0], 3), round(c[0][1], 3)),
                                      (round(c[-1][0], 3), round(c[-1][1], 3))])

                from collections import Counter
                node_degrees = Counter(endpoints)

                to_keep = []
                removed_any = False

                for cl in unique_cl:
                    c = list(cl['line'].coords)
                    p1_r = (round(c[0][0], 3), round(c[0][1], 3))
                    p2_r = (round(c[-1][0], 3), round(c[-1][1], 3))
                    L = cl['line'].length

                    if (node_degrees[p1_r] == 1 or node_degrees[p2_r] == 1) and L < trim_tol:
                        removed_any = True
                    else:
                        to_keep.append(cl)

                unique_cl = to_keep
                if not removed_any:
                    break

            return unique_cl

        def weld_vertices(centerlines, weld_tol=1.0):
            """1mm 이내로 인접한 노드들을 강제로 하나의 좌표로 병합(Weld)하여 틈새를 0으로 만듦"""
            endpoints = []
            for cl in centerlines:
                c = list(cl['line'].coords)
                endpoints.extend([c[0], c[-1]])

            # 인접 노드들을 하나의 대표 좌표로 묶는 딕셔너리 생성
            welded_nodes = {}
            for pt in endpoints:
                found = False
                for w_pt in welded_nodes:
                    if np.hypot(pt[0] - w_pt[0], pt[1] - w_pt[1]) <= weld_tol:
                        welded_nodes[pt] = w_pt
                        found = True
                        break
                if not found:
                    welded_nodes[pt] = pt

            new_cl = []
            for cl in centerlines:
                c = list(cl['line'].coords)
                p_s = welded_nodes.get(c[0], c[0])
                p_e = welded_nodes.get(c[-1], c[-1])

                # 병합 후 시작점과 끝점이 같아진(길이가 0이 된) 선분은 제외
                if np.hypot(p_s[0] - p_e[0], p_s[1] - p_e[1]) > 0.1:
                    new_item = cl.copy()
                    c[0] = p_s
                    c[-1] = p_e
                    new_item['line'] = LineString(c)
                    new_cl.append(new_item)

            return new_cl

        def heal_collinear_centerlines(centerlines, max_gap=400.0, angle_tol=2.0, align_tol=15.0):
            """
            정렬된 1D 중심선들 중, 동일 선상에 있으나 끊어져 있는 부재(Gap)를 찾아
            새로운 선분(Bridge)으로 잇습니다.
            """
            meta = []
            for i, cl in enumerate(centerlines):
                c = list(cl['line'].coords)
                if len(c) < 2: continue
                p1, p2 = np.array(c[0]), np.array(c[-1])
                v = p2 - p1
                L = np.linalg.norm(v)
                if L < 1e-6: continue
                ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
                meta.append({'cl': cl, 'p1': p1, 'p2': p2, 'ang': ang, 'v': v / L, 'L': L})

            # 1. 각도와 수직 거리를 기반으로 동일 선상 그룹화
            groups = []
            for m in meta:
                placed = False
                for g in groups:
                    g_ref = g[0]
                    # 각도 차이 확인
                    ang_diff = min(abs(m['ang'] - g_ref['ang']), 180 - abs(m['ang'] - g_ref['ang']))
                    if ang_diff > angle_tol: continue

                    # 두 직선 간의 평행 간격(수직 거리) 확인
                    v_ref = g_ref['p2'] - g_ref['p1']
                    v_target = g_ref['p1'] - m['p1']
                    d = abs(v_ref[0] * v_target[1] - v_ref[1] * v_target[0]) / g_ref['L']
                    if d <= align_tol:
                        g.append(m)
                        placed = True
                        break
                if not placed:
                    groups.append([m])

            # 2. 그룹 내에서 틈새(gap) 찾아 이어주기
            bridges = []
            for g in groups:
                if len(g) < 2: continue
                dv = g[0]['v']
                segs = []
                # 벡터 방향으로 투영(Projection)하여 선분 정렬
                for m in g:
                    t1 = np.dot(m['p1'], dv)
                    t2 = np.dot(m['p2'], dv)
                    if t1 > t2:
                        segs.append({'t_min': t2, 't_max': t1, 'p_min': m['p2'], 'p_max': m['p1'], 'cl': m['cl']})
                    else:
                        segs.append({'t_min': t1, 't_max': t2, 'p_min': m['p1'], 'p_max': m['p2'], 'cl': m['cl']})

                segs.sort(key=lambda x: x['t_min'])

                # 정렬된 선분 사이의 간격 검사
                for i in range(len(segs) - 1):
                    gap = segs[i + 1]['t_min'] - segs[i]['t_max']
                    # 겹친 상태(gap < 0)가 아니고, 최대 허용 틈새(max_gap) 이내일 때 연결
                    if 0.1 < gap <= max_gap:
                        new_line = LineString([tuple(segs[i]['p_max']), tuple(segs[i + 1]['p_min'])])
                        # 양쪽 부재의 두께 평균을 물성치로 할당
                        thk = (segs[i]['cl'].get('thickness', 10.0) + segs[i + 1]['cl'].get('thickness', 10.0)) / 2.0
                        bridges.append({
                            'line': new_line,
                            'thickness': thk,
                            'type': segs[i]['cl'].get('type', 'unknown'),
                            'color': segs[i]['cl'].get('color', '#003087')
                        })

            return centerlines + bridges

        # ---------------------------------------------------------------

        try:
            progress.setLabelText("Extracting DXF and Creating Initial 1D Lines...")
            progress.setValue(10)
            QApplication.processEvents()

            # --- (기존 코드) Phase 1 진입 전 ---
            y_mins = []
            for l_seg in self.left_1999_segments:
                y_mins.append(l_seg.bounds[1] - 10.0 / 2.0)
            thickness_y_min = min(y_mins) if y_mins else 0.0

            c1102 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_1102_raw]
            c157 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_157]
            l1999s = [affinity.translate(l, yoff=-thickness_y_min) for l in self.left_1999_segments]

            c6001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_6001]
            c7001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_7001]
            c8001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_8001]
            c9001 = [affinity.translate(l, yoff=-thickness_y_min) for l in self.lines_9001]

            # 3. 외판(1999) 중심선 생성
            cl1999 = []

            cl1999 = []
            for ls in l1999s:
                if ls.length > 50.0:
                    cl1999.append({'line': ls, 'thickness': 10.0, 'type': '1999', 'color': '#333333'})

            # ✨ 각도/그룹 무시하고 무조건 300mm 결합
            cl1999 = self.robust_heal_1999(cl1999, max_gap=300.0)

            s1102_raw, s157_raw = [], []
            for l in filter_short(c1102, 10.0): s1102_raw.extend(split_by_slope(l, at=5.0))
            for l in filter_short(c157, 10.0): s157_raw.extend(split_by_slope(l, at=5.0))

            f1102 = remove_overlapping(filter_short(s1102_raw, 50.0), dt=1.0)
            f157 = remove_overlapping(filter_short(s157_raw, 50.0), dt=1.0)

            # --- 내부 부재 보정 (150mm 일괄 적용) ---
            # heal_1102_collinear 대신 범용 함수인 heal_internal_collinear 사용
            h1102 = self.heal_internal_collinear(f1102, threshold_gap=150.0)
            h157 = self.heal_internal_collinear(f157, threshold_gap=150.0)  # 157 레이어도 150mm 틈새 보정 추가

            p1102 = match_pairs(h1102, max_dist=100.0, angle_tol=5.0, overlap_tolerance=5.0)
            p157 = match_pairs(h157, max_dist=100.0, angle_tol=5.0, overlap_tolerance=5.0)

            cl1102 = create_centerlines(p1102)
            for cl in cl1102: cl['type'] = '1102'
            cl157 = create_centerlines(p157)
            for cl in cl157: cl['type'] = '157'

            progress.setLabelText("Grouping and Aligning Internal Members...")
            progress.setValue(40)
            QApplication.processEvents()

            internal_cl = cl1102 + cl157
            self.aligned_internal = self.group_and_align_centerlines(internal_cl, tol_dist=150.0, tol_angle=1.5)

            all_cl = cl1999 + self.aligned_internal

            progress.setLabelText("Phase 0.5: Healing Collinear Gaps...")
            all_cl = heal_collinear_centerlines(all_cl, max_gap=400.0)

            # ✨ 1차 보정: 근접한 틈새(10mm 이내) 삐져나감 정리 및 병합
            progress.setLabelText("Phase 1: Healing Small Gaps (10mm)...")
            progress.setValue(60)
            all_cl = extend_and_trim_10mm(all_cl)
            all_cl.sort(key=lambda x: 1 if x.get('type') == '157' else 0)

            # ✨ 2차 보정: 원거리 틈새를 찾아 외판/교차점까지 확장 (Raycasting)
            progress.setLabelText("Phase 2: Short Raycasting (50mm)...")
            progress.setValue(65)
            all_cl, _ = raycast_extend(all_cl, max_dist=50.0)

            progress.setLabelText("Phase 2: Intermediate Topology Cleanup...")
            progress.setValue(75)
            all_cl = split_all_lines_at_intersections(all_cl)
            all_cl = weld_vertices(all_cl, weld_tol=1.0)
            all_cl = clean_topology(all_cl, trim_tol=100.0)  # 내부의 자잘한 꼬리 먼저 제거

            progress.setLabelText("Phase 3: Deep Raycasting for Decks...")
            progress.setValue(85)
            # 내부가 깔끔해진 상태에서 데크 끝단 등 진짜 뚫려있는 곳만 길게 뻗어나감
            all_cl, _ = raycast_extend(all_cl, max_dist=400.0)

            progress.setLabelText("Phase 3: Final Topology Cleanup...")
            progress.setValue(90)
            all_cl = split_all_lines_at_intersections(all_cl)
            all_cl = clean_topology(all_cl, trim_tol=300.0)
            all_cl = weld_vertices(all_cl, weld_tol=1.0)

            # ✨ 4차 정리: 맞닿은 선들을 노드별로 분할하여 위상(Topology) 확립
            progress.setLabelText("Phase 4: Topology Cleanup...")
            progress.setValue(90)
            all_cl = split_all_lines_at_intersections(all_cl)

            # ✨ 5차 최종 정리: 1.0mm 이내의 미세 틈새 노드 강제 병합(Weld)
            progress.setLabelText("Phase 5: Welding Vertices...")
            progress.setValue(98)
            all_cl = weld_vertices(all_cl, weld_tol=1.0)

            # ✨ 6차 최종 클린업: 이중 선분 제거 및 삐져나온 꼬리 트림 (새로 추가된 부분)
            progress.setLabelText("Phase 6: Removing Duplicates and Trimming...")
            progress.setValue(95)
            # trim_tol=50.0 이면 50mm 이하로 삐져나온 꼬리를 전부 잘라냅니다. (필요 시 수정 가능)
            all_cl = clean_topology(all_cl, trim_tol=100.0)

            self.hull_only_centerlines = [cl.copy() for cl in all_cl]

            progress.setLabelText("Processing Stiffeners (6001~9001)...")
            raw_stiffs = c6001 + c7001 + c8001 + c9001
            stiff_s_raw = []
            for l in filter_short(raw_stiffs, 10.0):
                stiff_s_raw.extend(split_by_slope(l, at=5.0))

            stiff_s = remove_overlapping(filter_short(stiff_s_raw, 20.0), dt=1.0)
            stiff_pairs = match_pairs(stiff_s, max_dist=50.0, angle_tol=20.0, overlap_tolerance=5.0)

            stiff_cl = create_continuous_stiffener_centerlines(stiff_pairs)
            for c in stiff_cl:
                c['type'] = 'stiffener'
                c['color'] = '#FF7F0E'  # 보강재를 도면에서 식별하기 위한 주황색 지정

            # 주 구조물(all_cl)에 보강재(stiff_cl) 합병
            all_cl.extend(stiff_cl)
            self.final_healed_centerlines = all_cl

            progress.setLabelText("Building Mathematical Graph...")
            self.build_graph()

            progress.setLabelText("Detecting Closed Cells (Loops)...")
            self.detect_closed_cells()

            progress.setLabelText("Executing Flowchart Algorithm...")
            self.execute_flowchart_algorithm()

            # ✨ 팝업창 대신 General Settings의 텍스트 상자에서 전단력 가져오기
            try:
                Vy_input_t = float(self.txt_vy.text().strip())
                # 1 ton(t) = 9806.65 N 으로 변환 (응력 및 전단류 단위 N/mm 연동을 위함)
                Vy_input_N = Vy_input_t * 9806.65

            except ValueError:
                # 임의의 값 대신 경고 팝업창을 띄우고 변환 프로세스 즉시 중단
                progress.close()
                self.is_processing = False
                self.btn_calc.setEnabled(True)
                self.btn_load.setEnabled(True)
                QMessageBox.warning(self, "입력 오류", "전단력(Total Vy)에는 유효한 숫자(t 단위)를 입력해야 합니다.\n계산을 중단합니다.")
                return

            progress.setLabelText("Calculating Determinate Shear Flow...")
            self.calculate_determinate_shear_flow(Vy_total=Vy_input_N)
            QApplication.processEvents()

            progress.setLabelText("Calculating Section Properties...")
            progress.setValue(99)
            QApplication.processEvents()

            total_area = 0.0
            sum_qx = 0.0
            segments_1d = []

            # 1. 전체 면적 및 1차 모멘트(Qx) 계산
            for cl in all_cl:
                coords = list(cl['line'].coords)
                thk = cl.get('thickness', 10.0)
                if thk <= 0: thk = 10.0

                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    dx, dy = x2 - x1, y2 - y1
                    L = np.hypot(dx, dy)
                    if L < 1e-6: continue

                    a = L * thk
                    yc = (y1 + y2) / 2.0  # 선분의 무게중심 Y좌표

                    total_area += a
                    sum_qx += a * yc
                    segments_1d.append((a, yc, dx, dy, L))

            # 2. 중립축(N.A) 및 평행축 정리를 이용한 단면 2차 모멘트(Ixx) 계산
            if total_area > 0:
                na_y = sum_qx / total_area
                ixx = 0.0
                ixxm = 0.0

                for a, yc, dx, dy, L in segments_1d:
                    # 국부 관성모멘트 (Local Ixx)
                    i_local = (a * (dy ** 2)) / 12.0
                    # 평행축 정리
                    ixx += i_local + a * ((yc - na_y) ** 2)
                ixxm = ixx / 1e12

                # 추후 전단응력 계산 등을 위해 클래스 변수에 저장
                self.calc_total_area = total_area
                self.calc_na_bl = na_y
                self.calc_ixx = ixxm

                calc_result_text = (
                    f"\n\n📊 [Section Properties Result]\n"
                    f"----------------------------------------\n"
                    f"- Total Area (단면적)    : {total_area:,.2f} mm²\n"
                    f"- N.A from Base (중립축) : {na_y:,.2f} mm\n"
                    f"- Moment of Inertia (I_xx): {ixxm:,.2e} m⁴\n"
                    f"----------------------------------------\n"
                )
            else:
                calc_result_text = "\n\n❌ 유효한 단면적이 없어 이너시아를 계산할 수 없습니다."

            # 3. 결과창 출력 텍스트 업데이트
            summary_text = (
                "✅ 1D Transformation & Full Healing Complete!\n\n"
                f"- Extracted Outer Hull Lines (1999): {len(cl1999)}\n"
                f"- Aligned Internal Lines (1102, 157): {len(self.aligned_internal)}\n"
                f"- Added Stiffeners (6001~9001): {len(stiff_cl)}\n"
                f"- Final Healed Elements: {len(all_cl)}"
                f"{calc_result_text}"
                f"{self.graph_summary}"  # ✨ 추가됨
            )
            self.result_box.setText(summary_text)

            self.is_calculated = True
            progress.setValue(100)
            self.refresh_ui()

        except Exception as e:
            import traceback
            self.result_box.setText(f"❌ Error:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            progress.close()
            self.is_processing = False
            self.btn_calc.setEnabled(True)
            self.btn_load.setEnabled(True)

        # =====================================================================
        # ✨ 신규 추가: 위상 노드 추출 (반단면 컷팅 & 선분 클릭 이벤트 포함)
        # =====================================================================
    def build_graph(self):
        from collections import defaultdict
        from shapely.geometry import box, LineString
        import numpy as np

        # ==================================================
        # 1. 중심선(Centerline) 컷팅 위치 계산 (가장 높은 지점의 X 기준)
        # ==================================================
        max_y_global = -float('inf')
        highest_xs = []
        for cl in self.final_healed_centerlines:
            if cl.get('type') in ['6001', '7001', '8001', '9001', 'stiffener']:
                continue
            for cx, cy in cl['line'].coords:
                if cy > max_y_global + 1e-3:
                    max_y_global = cy
                    highest_xs = [cx]
                elif abs(cy - max_y_global) <= 1e-3:
                    highest_xs.append(cx)

        self.x_cut = sum(highest_xs) / len(highest_xs) if highest_xs else 0.0

        # 반단면 컷팅용 박스 생성 (x_cut 기준으로 좌측만 남김, 오차 허용치 +0.5)
        keep_box = box(-9999999.0, -9999999.0, self.x_cut + 0.5, 9999999.0)

        raw_nodes = {}
        raw_edges = []
        node_map = {}
        nid_counter = 1

        def get_nid(pt):
            nonlocal nid_counter
            for ept, eid in node_map.items():
                # 1.0mm 이내의 점들은 동일 노드로 취급
                if np.hypot(ept[0] - pt[0], ept[1] - pt[1]) < 1.0:
                    return eid
            node_map[pt] = nid_counter
            raw_nodes[nid_counter] = pt
            nid_counter += 1
            return node_map[pt]

        # ==================================================
        # 2. 중심선 컷팅 및 초기 선분(Edge) 생성
        # ==================================================
        for cl in self.final_healed_centerlines:
            if cl.get('type') in ['6001', '7001', '8001', '9001', 'stiffener']:
                continue

            geom = cl['line']

            # ✨ 중심선(keep_box)을 기준으로 컷팅! (여기서 컷팅된 지점이 자연스럽게 끝점 좌표가 됨)
            clipped = geom.intersection(keep_box)
            if clipped.is_empty: continue

            geoms = [clipped] if clipped.geom_type == 'LineString' else \
                list(clipped.geoms) if clipped.geom_type == 'MultiLineString' else []

            for g in geoms:
                if g.length < 1.0: continue  # 너무 짧은 찌꺼기 무시
                c = list(g.coords)

                # 끝점이 get_nid를 통과하며 중심선에 잘린 위치에도 정확히 노드가 생성됨
                u, v = get_nid(c[0]), get_nid(c[-1])
                raw_edges.append({
                    'u': u, 'v': v,
                    'coords': c,
                    'thickness': cl.get('thickness', 10.0),
                    'type': cl.get('type', 'unknown')
                })

        # ==================================================
        # 3. 인접 노드(Degree 2) 병합 알고리즘 (곡선 경로 보존)
        # ==================================================
        while True:
            node_to_edges = defaultdict(list)
            for i, e in enumerate(raw_edges):
                if e is None: continue
                node_to_edges[e['u']].append(i)
                if e['u'] != e['v']:
                    node_to_edges[e['v']].append(i)

            merged_any = False
            for nid, edge_indices in node_to_edges.items():
                if len(edge_indices) == 2:
                    # ✨ 아주 중요한 예외 처리: 중심선(x_cut)에 위치한 노드는 병합하여 삭제하지 않고 구조적 경계로 무조건 보존
                    node_pt = raw_nodes[nid]
                    if abs(node_pt[0] - self.x_cut) <= 1.5:
                        continue

                    e1_idx, e2_idx = edge_indices[0], edge_indices[1]
                    if e1_idx == e2_idx: continue
                    e1, e2 = raw_edges[e1_idx], raw_edges[e2_idx]

                    # 타입이나 두께가 다르면 교차점으로 인식하여 병합 안 함
                    if abs(e1['thickness'] - e2['thickness']) > 0.1 or e1['type'] != e2['type']:
                        continue

                    c1, c2 = e1['coords'], e2['coords']

                    if e1['v'] == nid:
                        c1_ordered, u_new = c1, e1['u']
                    else:
                        c1_ordered, u_new = c1[::-1], e1['v']

                    if e2['u'] == nid:
                        c2_ordered, v_new = c2[1:], e2['v']
                    else:
                        c2_ordered, v_new = c2[::-1][1:], e2['u']

                    merged_coords = c1_ordered + c2_ordered

                    raw_edges[e1_idx] = {
                        'u': u_new, 'v': v_new,
                        'coords': merged_coords,
                        'thickness': e1['thickness'],
                        'type': e1['type']
                    }
                    raw_edges[e2_idx] = None
                    merged_any = True
                    break

            if not merged_any: break

        # ==================================================
        # 4. 최종 그래프 구조화
        # ==================================================
        self.graph_nodes = {}
        self.graph_edges = []
        final_raw_edges = [e for e in raw_edges if e is not None]

        active_nids = set()
        for e in final_raw_edges:
            active_nids.update([e['u'], e['v']])

        for nid in active_nids:
            self.graph_nodes[nid] = raw_nodes[nid]

        for i, e in enumerate(final_raw_edges):
            geom = LineString(e['coords'])
            self.graph_edges.append({
                'id': i,
                'start_node': e['u'],
                'start_coord': self.graph_nodes[e['u']],
                'end_node': e['v'],
                'end_coord': self.graph_nodes[e['v']],
                'thickness': e['thickness'],
                'length': geom.length,
                'geometry': geom,
                'type': e['type']
            })

        self.graph_summary = (
            f"\n\n📊 [Half-Section Graphification Result]\n"
            f"----------------------------------------\n"
            f"- 중심선 컷팅 X 좌표 : {self.x_cut:.2f} mm\n"
            f"- 유효 노드 (Nodes)  : {len(self.graph_nodes)} 개\n"
            f"- 생성된 선분 (Edges)  : {len(self.graph_edges)} 개\n"
            f"----------------------------------------\n"
            )

    def detect_closed_cells(self):
        """Shapely를 활용하여 단면 내의 폐쇄된 방(Cell)과 공유 격벽을 엄격하게 탐색합니다."""
        from shapely.ops import polygonize, unary_union

        self.cells_info = []
        self.edge_to_cells = {e['id']: [] for e in self.graph_edges}

        # 1. 1D 선분 기하학 추출 및 강제 결합 (미세한 틈새로 인해 루프가 인식되지 않는 현상 원천 차단)
        lines = [e['geometry'] for e in self.graph_edges]
        merged_lines = unary_union(lines)

        # 2. 다각형(Cell) 생성
        polygons = list(polygonize(merged_lines))

        for i, poly in enumerate(polygons):
            cell_id = i + 1
            cell_area = poly.area
            cell_edges = []

            # 3. 폴리곤 경계(Boundary)에 포함되는 선분 식별
            bound = poly.boundary
            for eid, e in enumerate(self.graph_edges):
                geom = e['geometry']
                # 선분의 중간점(centroid)을 기준으로 검사하여 인식률 극대화 (허용 오차 1e-2)
                if bound.distance(geom.centroid) < 1e-2 or bound.distance(geom) < 1e-2:
                    cell_edges.append(eid)
                    self.edge_to_cells[eid].append(cell_id)

            self.cells_info.append({
                'cell_id': cell_id,
                'polygon': poly,
                'area': cell_area,
                'edges': cell_edges
            })

        # =====================================================================
        # ✨ 첨부된 순서도(Flowchart)의 완벽한 논리 이식
        # =====================================================================

    def execute_flowchart_algorithm(self):
        """순서도에 명시된 Free Node Pruning과 Loop Detection을 엄격하게 수행합니다."""

        # 1. 시뮬레이션용 복사본 (알고리즘 정상 종료를 위해 파괴될 가상 그래프)
        sim_edges = {e['id']: {'u': e['start_node'], 'v': e['end_node']} for e in self.graph_edges}

        # 2. 실제 보존될 최종 그래프 (더미 노드 생성 및 부재 보존용)
        working_edges = {e['id']: {'u': e['start_node'], 'v': e['end_node']} for e in self.graph_edges}

        slit_nodes_ids = []
        self.cut_edges_info = []
        self.remaining_edges_info = {}

        def get_degrees(edges_dict):
            deg = {}
            for e in edges_dict.values():
                deg[e['u']] = deg.get(e['u'], 0) + 1
                deg[e['v']] = deg.get(e['v'], 0) + 1
            return deg

        while True:
            # [Phase 1: 좌측 순서도] 자유 노드 가지치기 (가상 그래프에서만 삭제)
            while True:
                deg = get_degrees(sim_edges)
                free_nodes = [n for n, d in deg.items() if d == 1]
                if not free_nodes: break
                i = free_nodes[0]

                while True:
                    m = None
                    for eid, e in sim_edges.items():
                        if e['u'] == i or e['v'] == i:
                            m = eid
                            break
                    if m is None: break

                    j = sim_edges[m]['v'] if sim_edges[m]['u'] == i else sim_edges[m]['u']
                    deg_j = get_degrees(sim_edges).get(j, 0)

                    # 시뮬레이션 그래프에서는 완전 삭제하여 무한 루프 방지
                    del sim_edges[m]

                    if deg_j == 1:
                        break
                    elif deg_j >= 3:
                        break
                    else:
                        i = j

            # [Phase 2: 우측 순서도] 루프 탐지 및 실제 슬릿(더미 노드) 생성
            deg = get_degrees(sim_edges)
            remaining_nodes = [n for n, d in deg.items() if d > 0]

            if not remaining_nodes: break

            i = remaining_nodes[0]
            visited_nodes = {i}
            visited_members = set()

            while True:
                attached_edges = [eid for eid, e in sim_edges.items() if
                                  (e['u'] == i or e['v'] == i) and eid not in visited_members]
                if not attached_edges: break

                m = attached_edges[0]
                visited_members.add(m)

                j = sim_edges[m]['v'] if sim_edges[m]['u'] == i else sim_edges[m]['u']

                if j in visited_nodes:
                    # ✨ 루프 발견! 기록 및 실제 그래프(working_edges) 분리 작업 수행
                    slit_nodes_ids.append(j)

                    # 1. 원본 노드와 동일한 좌표를 갖는 새로운 식별 번호(Dummy Node) 생성
                    new_nid = max(self.graph_nodes.keys()) + 1
                    self.graph_nodes[new_nid] = self.graph_nodes[j]

                    # 2. 실제 보존될 그래프의 간선 끝점을 새로운 더미 노드로 갈아끼워 분리
                    if working_edges[m]['u'] == j:
                        working_edges[m]['u'] = new_nid
                    else:
                        working_edges[m]['v'] = new_nid

                    # 3. 팝업창 가시화를 위한 정보 저장
                    self.cut_edges_info.append({
                        'edge_id': m,
                        'original_nid': j,
                        'dummy_nid': new_nid
                    })

                    # 🚨 시뮬레이션 그래프(sim_edges)에서는 해당 간선을 파괴하여 탐색 종료 유도
                    del sim_edges[m]
                    break
                else:
                    visited_nodes.add(j)
                    i = j

        self.flowchart_slit_nodes = list(set(slit_nodes_ids))
        self.remaining_edges_info = working_edges  # 더미 노드가 완벽히 분리된 최종 그래프

    def on_edge_click(self, event):
        """그래프 선분 클릭 시 정보를 표시하는 이벤트 핸들러"""
        if event.artist and hasattr(event.artist, 'get_gid'):
            gid = event.artist.get_gid()
            if gid is not None and str(gid).startswith("edge_"):
                edge_id = int(str(gid).split("_")[1])
                edge = self.graph_edges[edge_id]

                info_text = (
                    f"🔹 [선분 상세 정보]\n\n"
                    f" - 부재 타입 : {edge['type']}\n"
                    f" - 시작 노드 : Node {edge['start_node']} (X: {edge['start_coord'][0]:.1f}, Y: {edge['start_coord'][1]:.1f})\n"
                    f" - 끝 노드   : Node {edge['end_node']} (X: {edge['end_coord'][0]:.1f}, Y: {edge['end_coord'][1]:.1f})\n"
                    f" - 두께(t)   : {edge['thickness']} mm\n"
                    f" - 길이(L)   : {edge['length']:,.2f} mm\n"
                )

                # ✨ 300mm 단위로 쪼개진 전단류 및 전단응력 계산 결과 상세 정보 추가
                if hasattr(self, 'edge_q_results') and edge_id in self.edge_q_results:
                    chunks = self.edge_q_results[edge_id]
                    q_s_unit = chunks[0]['q_start_unit']
                    q_e_unit = chunks[-1]['q_end_unit']

                    user_vy = getattr(self, 'user_Vy_total', 1000000.0)
                    q_s_actual = q_s_unit * user_vy
                    q_e_actual = q_e_unit * user_vy

                    # 전단류(N/mm) / 두께(mm) = 전단응력(MPa)
                    thk = edge['thickness']
                    tau_s_actual = q_s_actual / thk
                    tau_e_actual = q_e_actual / thk

                    if hasattr(self, 'edge_to_cells') and edge_id in self.edge_to_cells:
                        belonging_cells = self.edge_to_cells[edge_id]
                        if len(belonging_cells) == 0:
                            shared_text = "일반 개단면 부재 (폐루프 미포함)"
                        elif len(belonging_cells) == 1:
                            shared_text = f"외판/독립 격벽 (Cell {belonging_cells[0]} 단독 소속)"
                        else:
                            shared_text = f"🔥 공유 격벽 (Shared Web) - Cell {belonging_cells} 사이 연결"

                        info_text += (
                            f"\n\n🟩 [위상수학적 셀(Cell) 정보]\n"
                            f" - 소속 상태 : {shared_text}\n"
                        )

                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, f"Edge ID: {edge_id}", info_text)

    def on_slit_node_click(self, event):
        """슬릿 노드(별모양) 클릭 시 위상 팝업창을 띄우는 이벤트 핸들러"""
        if event.artist and hasattr(event.artist, 'get_gid'):
            gid = event.artist.get_gid()
            if gid is not None and str(gid).startswith("slit_"):
                nid = int(str(gid).split("_")[1])
                # 팝업 띄우기
                dialog = SlitViewerDialog(self, nid)
                dialog.exec()

    def refresh_ui(self):
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except:
                pass
            self.cbar = None

        self.fig1.clear()
        self.fig2.clear()
        ax1, ax2 = self.fig1.add_subplot(111), self.fig2.add_subplot(111)

        if self.is_calculated:
            # [도면 1: 최종 1D 형상 뷰]
            if hasattr(self, 'final_healed_centerlines'):
                for cl in self.final_healed_centerlines:
                    lo = cl['line']
                    c_type = cl.get('type')
                    final_color = '#000000' if c_type == '1999' else ('#FF7F0E' if c_type == 'stiffener' else '#003087')
                    thk = cl.get('thickness', 10.0)
                    visual_thk = thk if thk >= 50.0 else 50.0

                    if visual_thk > 0:
                        try:
                            poly = lo.buffer(visual_thk / 2.0, cap_style=2)
                            if poly.geom_type == 'Polygon':
                                ax1.fill(*poly.exterior.xy, color=final_color, alpha=0.3, zorder=9, edgecolor='none')
                            elif poly.geom_type == 'MultiPolygon':
                                for p in poly.geoms:
                                    ax1.fill(*p.exterior.xy, color=final_color, alpha=0.3, zorder=9, edgecolor='none')
                        except:
                            pass
                    ax1.plot(*lo.xy, color=final_color, linewidth=2.0, alpha=0.9, zorder=10)

            # ✨ [도면 2: 수직 막대그래프형 전단응력 다이어그램]
            if hasattr(self, 'graph_edges'):
                if hasattr(self, 'cells_info') and self.cells_info:
                    for cinfo in self.cells_info:
                        poly = cinfo['polygon']
                        cid = cinfo['cell_id']

                        cx, cy = poly.centroid.x, poly.centroid.y
                        ax2.annotate(f"Cell {cid}", (cx, cy),
                                     color='black', weight='bold', fontsize=12,
                                     ha='center', va='center', zorder=25,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9, lw=1.5))

                if hasattr(self, 'edge_q_results') and self.edge_q_results:
                    import matplotlib.colors as mcolors
                    import matplotlib.cm as cm

                    max_tau = 1e-9
                    user_vy = getattr(self, 'user_Vy_total', 1000000.0)

                    for eid, sub_list in self.edge_q_results.items():
                        thk = self.graph_edges[eid].get('thickness', 10.0)
                        if thk < 0.1: thk = 0.1

                        for chunk in sub_list:
                            tau_start = abs(chunk['q_start_unit'] * user_vy / thk)
                            tau_end = abs(chunk['q_end_unit'] * user_vy / thk)
                            max_tau = max(max_tau, tau_start, tau_end)

                    nodes_arr = np.array(list(self.graph_nodes.values()))
                    if len(nodes_arr) > 0:
                        max_dim = max(nodes_arr[:, 0].max() - nodes_arr[:, 0].min(),
                                      nodes_arr[:, 1].max() - nodes_arr[:, 1].min())
                        cx_hull, cy_hull = np.mean(nodes_arr, axis=0)
                    else:
                        max_dim = 10000.0
                        cx_hull, cy_hull = 0.0, 0.0

                    visual_scale = (max_dim * 0.05) / max_tau if max_tau > 1e-9 else 1.0
                    cmap = cm.turbo
                    norm = mcolors.Normalize(vmin=0, vmax=max_tau)

                    for eid, edge_data in enumerate(self.graph_edges):
                        res_list = self.edge_q_results.get(eid)
                        thk = edge_data.get('thickness', 10.0)
                        if thk < 0.1: thk = 0.1

                        if res_list:
                            geom_c = list(edge_data['geometry'].coords)
                            mid_pt = geom_c[len(geom_c) // 2]
                            v_inward = np.array([cx_hull - mid_pt[0], cy_hull - mid_pt[1]])

                            dx = geom_c[-1][0] - geom_c[0][0]
                            dy = geom_c[-1][1] - geom_c[0][1]
                            base_n = np.array([-dy, dx])

                            base_tg = np.array([dx, dy])
                            if np.linalg.norm(base_tg) > 1e-6:
                                base_tg = base_tg / np.linalg.norm(base_tg)
                            else:
                                base_tg = np.array([1.0, 0.0])

                            flip_n = -1.0 if np.dot(base_n, v_inward) < 0 else 1.0

                            for chunk in res_list:
                                sub_geom = chunk['geom']
                                is_fwd = chunk.get('is_forward', True)

                                tau_s_signed = (chunk['q_start_unit'] * user_vy) / thk
                                tau_e_signed = (chunk['q_end_unit'] * user_vy) / thk

                                c = np.array(sub_geom.coords)
                                num_pts = len(c)

                                if is_fwd:
                                    taus_signed = np.linspace(tau_s_signed, tau_e_signed, num_pts)
                                else:
                                    taus_signed = np.linspace(tau_e_signed, tau_s_signed, num_pts)

                                taus = np.abs(taus_signed)

                                n_vecs = np.zeros_like(c)
                                for p_idx in range(num_pts):
                                    if num_pts == 2:
                                        tg = c[1] - c[0]
                                    elif p_idx == 0:
                                        tg = c[1] - c[0]
                                    elif p_idx == num_pts - 1:
                                        tg = c[-1] - c[-2]
                                    else:
                                        tg = c[p_idx + 1] - c[p_idx - 1]

                                    tg_len = np.linalg.norm(tg)
                                    if tg_len > 1e-6:
                                        tg = tg / tg_len
                                    else:
                                        tg = base_tg

                                    n_vecs[p_idx] = np.array([-tg[1], tg[0]]) * flip_n

                                top_pts = c + n_vecs * (taus[:, np.newaxis] * visual_scale)

                                tau_mid_abs = abs((tau_s_signed + tau_e_signed) / 2.0)
                                face_color = cmap(norm(tau_mid_abs))

                                poly_x = list(c[:, 0]) + list(top_pts[::-1, 0])
                                poly_y = list(c[:, 1]) + list(top_pts[::-1, 1])

                                # ✨ 1. 테두리나 해치선 없이 반투명한 색상으로만 칠하여 자연스러운 겹침 유도
                                ax2.fill(poly_x, poly_y, color=face_color, alpha=0.6, edgecolor='none', zorder=8)

                            # ✨ 2. 조각(Chunk)을 그리는 내부 루프를 빠져나와, 원본 부재 전체의 선을 한 번에 그림 (끊김 원천 차단)
                            geom = edge_data['geometry']
                            ax2.plot(*geom.xy, color='black', linewidth=1.5, zorder=10, picker=5, gid=f"edge_{eid}")

                        else:
                            geom = edge_data['geometry']
                            ax2.plot(*geom.xy, color='dodgerblue', linewidth=2.5, alpha=0.8,
                                     zorder=10, picker=5, gid=f"edge_{eid}")

                    cut_edge_ids = [c['edge_id'] for c in self.cut_edges_info] if hasattr(self,
                                                                                          'cut_edges_info') else []
                    for eid in cut_edge_ids:
                        if eid < len(self.graph_edges):
                            geom = self.graph_edges[eid]['geometry']
                            ax2.plot(*geom.xy, color='#BDC3C7', linewidth=2.5, linestyle='--', zorder=5)

                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    self.cbar = self.fig2.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)

                    user_vy = getattr(self, 'user_Vy_total', 0)
                    label_text = f'Shear Stress |τ_d| (MPa)\n[Total Vy = {user_vy:,.0f} N 적용]'
                    self.cbar.set_label(label_text, fontweight='bold', fontsize=10)

                else:
                    for edge in self.graph_edges:
                        geom = edge['geometry']
                        ax2.plot(*geom.xy, color='dodgerblue', linewidth=2.5, alpha=0.8, zorder=10, picker=5,
                                 gid=f"edge_{edge['id']}")

                node_xs = [pt[0] for pt in self.graph_nodes.values()]
                node_ys = [pt[1] for pt in self.graph_nodes.values()]
                ax2.scatter(node_xs, node_ys, color='red', s=40, zorder=20, edgecolors='black')

                for nid, pt in self.graph_nodes.items():
                    ax2.annotate(str(nid), (pt[0], pt[1]), xytext=(4, 4), textcoords='offset points',
                                 color='black', fontsize=9, fontweight='bold', zorder=25)

                if hasattr(self, 'flowchart_slit_nodes') and self.flowchart_slit_nodes:
                    for i, nid in enumerate(self.flowchart_slit_nodes):
                        if nid in self.graph_nodes:
                            pt = self.graph_nodes[nid]
                            label = 'Slit Position (Cut Node)' if i == 0 else ""
                            ax2.scatter(pt[0], pt[1], color='lime', marker='*', s=250, zorder=30, edgecolors='black',
                                        picker=5, gid=f"slit_{nid}", label=label)
                    ax2.legend(loc='upper right')

        else:
            if hasattr(self, 'raw_1999_lines') and self.raw_1999_lines:
                for ls in self.raw_1999_lines:
                    ax1.plot(*ls.xy, color='black', lw=1.2, alpha=0.8, zorder=5)
                    ax2.plot(*ls.xy, color='black', lw=1.2, alpha=0.8, zorder=5)

        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.grid(True, lw=0.3, linestyle=':')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{-x:g}"))

        self.can1.draw()
        self.can2.draw()

    def get_topological_nodes(self, centerlines, tolerance=1.0):
        """
        교차점 노드를 추출하되, '일직선 상에 있으면서 두께 변화가 없는' 불필요한 노드는 탈락시킵니다.
        (6001~9001 보강재 레이어는 노드 생성에서 제외)
        """
        from collections import defaultdict
        import numpy as np

        node_map = defaultdict(list)
        # 1. 좌표별 연결된 선분 정보 기록
        for cl in centerlines:
            # ✨ 보강재 레이어(6001~9001, stiffener)는 노드 추출 대상에서 완전히 제외합니다.
            if cl.get('type') in ['6001', '7001', '8001', '9001', 'stiffener']:
                continue

            coords = list(cl['line'].coords)
            if len(coords) < 2: continue
            for pt in [tuple(coords[0]), tuple(coords[-1])]:
                found = False
                for existing_pt in node_map.keys():
                    if np.linalg.norm(np.array(pt) - np.array(existing_pt)) < tolerance:
                        node_map[existing_pt].append(cl)
                        found = True
                        break
                if not found:
                    node_map[pt].append(cl)

        final_nodes = []

        # 바깥쪽으로 향하는 방향 벡터를 구하는 내부 함수
        def get_outward_vector(cl, p_ref):
            c = list(cl['line'].coords)
            p_ref_arr = np.array(p_ref)
            d_start = np.linalg.norm(np.array(c[0]) - p_ref_arr)
            d_end = np.linalg.norm(np.array(c[-1]) - p_ref_arr)

            if d_start < d_end:
                v = np.array(c[1]) - np.array(c[0])
            else:
                v = np.array(c[-2]) - np.array(c[-1])

            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-6 else np.array([0, 0])

        # 2. 조건에 따른 노드 필터링 및 탈락
        for pt, connected_cls in node_map.items():
            if len(connected_cls) >= 3:
                final_nodes.append(pt)

            elif len(connected_cls) == 2:
                cl1, cl2 = connected_cls[0], connected_cls[1]
                t1 = cl1.get('thickness', 10.0)
                t2 = cl2.get('thickness', 10.0)

                if abs(t1 - t2) > 1e-3:
                    final_nodes.append(pt)
                    continue

                v1 = get_outward_vector(cl1, pt)
                v2 = get_outward_vector(cl2, pt)
                cos_theta = np.dot(v1, v2)

                if cos_theta > -0.996:
                    final_nodes.append(pt)
                else:
                    pass

        return final_nodes

    def calculate_determinate_shear_flow(self, Vy_total=1000000.0):
        """대칭 단면 1/mm 단위 정정전단류 계산 (빌지 곡선 보존 및 300mm 분할 반복)"""
        nodes = self.graph_nodes
        edges = self.remaining_edges_info
        from shapely.ops import substring

        # 1. 단면적 및 중립축(NA_y) 정밀 계산 (곡선 기하학 직접 반영)
        total_area = 0.0
        sum_qx = 0.0
        for eid, e in edges.items():
            geom = self.graph_edges[eid]['geometry']
            thk = self.graph_edges[eid]['thickness']
            L = geom.length
            if L < 1e-6: continue
            a = L * thk
            yc = geom.centroid.y
            total_area += a
            sum_qx += a * yc

        if total_area < 1e-6: return
        na_y = sum_qx / total_area

        # 2. I_xx 정밀 계산 (곡선을 50mm 단위로 미세 분할하여 평행축 정리 누적)
        ixx = 0.0
        for eid, e in edges.items():
            geom = self.graph_edges[eid]['geometry']
            thk = self.graph_edges[eid]['thickness']
            num_chunks = max(1, int(geom.length / 50.0))
            chunk_len = geom.length / num_chunks
            for i in range(num_chunks):
                sub = substring(geom, i * chunk_len, (i + 1) * chunk_len)
                a = sub.length * thk
                yc = sub.centroid.y
                c = list(sub.coords)
                dy = c[-1][1] - c[0][1]
                i_local = a * (dy ** 2) / 12.0
                ixx += i_local + a * (yc - na_y) ** 2

        if ixx == 0: return

        # 3. 순서도 초기화 단계
        from collections import defaultdict
        nm = defaultdict(int)
        for e in edges.values():
            nm[e['u']] += 1
            nm[e['v']] += 1

        vn = defaultdict(int)
        vm = {eid: 0 for eid in edges}

        node_flows_unit = defaultdict(dict)
        self.edge_q_results = {}

        # 4. 순서도 기반 경로 탐색 및 300mm 분할 누적 계산
        while True:
            # [Phase 1] 자유 노드
            path_started = False
            for n, deg in nm.items():
                if deg == 1 and vn[n] == 0:
                    i_curr = n
                    vn[i_curr] += 1
                    q_curr_unit = 0.0
                    path_started = True

                    while True:
                        m = None
                        for eid, e in edges.items():
                            if vm[eid] == 0 and (e['u'] == i_curr or e['v'] == i_curr):
                                m = eid
                                break
                        if m is None: break

                        j_next = edges[m]['v'] if edges[m]['u'] == i_curr else edges[m]['u']

                        # 곡선 기하학 및 진행 방향 판별
                        geom = self.graph_edges[m]['geometry']
                        thk = self.graph_edges[m]['thickness']
                        coord_u = nodes[i_curr]
                        geom_start = geom.coords[0]
                        geom_end = geom.coords[-1]

                        dist_to_start = np.hypot(coord_u[0] - geom_start[0], coord_u[1] - geom_start[1])
                        dist_to_end = np.hypot(coord_u[0] - geom_end[0], coord_u[1] - geom_end[1])
                        is_forward = dist_to_start < dist_to_end

                        # 300mm 단위로 곡선 경로 분할 계산
                        L_total = geom.length
                        num_chunks = max(1, int(np.ceil(L_total / 300.0)))

                        sub_results = []
                        for k in range(num_chunks):
                            if is_forward:
                                d1 = k * (L_total / num_chunks)
                                d2 = (k + 1) * (L_total / num_chunks)
                            else:
                                d1 = L_total - (k + 1) * (L_total / num_chunks)
                                d2 = L_total - k * (L_total / num_chunks)

                            sub_geom = substring(geom, min(d1, d2), max(d1, d2))
                            if sub_geom.length < 1e-6: continue

                            A_sub = sub_geom.length * thk
                            y_bar = sub_geom.centroid.y - na_y
                            dS_z = A_sub * y_bar

                            dq_unit = - (0.5 / ixx) * dS_z
                            q_next_unit = q_curr_unit + dq_unit

                            sub_results.append({
                                'geom': sub_geom,
                                'q_start_unit': q_curr_unit,
                                'q_end_unit': q_next_unit,
                                'is_forward': is_forward  # ✨ 렌더링 방향 일치를 위해 플래그 저장 추가
                            })
                            q_curr_unit = q_next_unit

                        self.edge_q_results[m] = sub_results
                        node_flows_unit[j_next][m] = q_curr_unit

                        vm[m] += 1
                        vn[j_next] += 1
                        i_curr = j_next

                        if nm[j_next] == 1 or nm[j_next] >= 3:
                            break

            # [Phase 2] 교차점 슈퍼포지션
            if not path_started:
                bridge_started = False
                for n, deg in nm.items():
                    if vn[n] == deg - 1 and deg >= 3:
                        m_unvisited = None
                        for eid, e in edges.items():
                            if vm[eid] == 0 and (e['u'] == n or e['v'] == n):
                                m_unvisited = eid
                                break

                        if m_unvisited is not None:
                            i_curr = n
                            vn[i_curr] += 1
                            q_curr_unit = sum(node_flows_unit[i_curr].values())
                            bridge_started = True

                            while True:
                                m = None
                                for eid, e in edges.items():
                                    if vm[eid] == 0 and (e['u'] == i_curr or e['v'] == i_curr):
                                        m = eid
                                        break
                                if m is None: break

                                j_next = edges[m]['v'] if edges[m]['u'] == i_curr else edges[m]['u']

                                geom = self.graph_edges[m]['geometry']
                                thk = self.graph_edges[m]['thickness']
                                coord_u = nodes[i_curr]
                                geom_start = geom.coords[0]
                                geom_end = geom.coords[-1]

                                dist_to_start = np.hypot(coord_u[0] - geom_start[0], coord_u[1] - geom_start[1])
                                dist_to_end = np.hypot(coord_u[0] - geom_end[0], coord_u[1] - geom_end[1])
                                is_forward = dist_to_start < dist_to_end

                                L_total = geom.length
                                num_chunks = max(1, int(np.ceil(L_total / 300.0)))

                                sub_results = []
                                for k in range(num_chunks):
                                    if is_forward:
                                        d1 = k * (L_total / num_chunks)
                                        d2 = (k + 1) * (L_total / num_chunks)
                                    else:
                                        d1 = L_total - (k + 1) * (L_total / num_chunks)
                                        d2 = L_total - k * (L_total / num_chunks)

                                    sub_geom = substring(geom, min(d1, d2), max(d1, d2))
                                    if sub_geom.length < 1e-6: continue

                                    A_sub = sub_geom.length * thk
                                    y_bar = sub_geom.centroid.y - na_y
                                    dS_z = A_sub * y_bar

                                    dq_unit = - (0.5 / ixx) * dS_z
                                    q_next_unit = q_curr_unit + dq_unit

                                    sub_results.append({
                                        'geom': sub_geom,
                                        'q_start_unit': q_curr_unit,
                                        'q_end_unit': q_next_unit,
                                        'is_forward': is_forward  # ✨ 렌더링 방향 일치를 위해 플래그 저장 추가
                                    })
                                    q_curr_unit = q_next_unit

                                self.edge_q_results[m] = sub_results
                                node_flows_unit[j_next][m] = q_curr_unit

                                vm[m] += 1
                                vn[j_next] += 1
                                i_curr = j_next

                                if nm[j_next] == 1 or nm[j_next] >= 3:
                                    break

                if not bridge_started:
                    break

        self.user_Vy_total = Vy_total


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UltimateShipAnalyzer()
    win.show()
    sys.exit(app.exec())
    

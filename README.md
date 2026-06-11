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
                               QInputDialog, QMessageBox, QProgressDialog, QRadioButton, QDialog)
from PySide6.QtGui import QTextCursor, QIcon
from PySide6.QtCore import Qt

from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union, polygonize, split, nearest_points, snap, substring
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

        for eid, e in self.main_app.remaining_edges_info.items():
            if e['u'] == self.slit_nid or e['v'] == self.slit_nid:
                geom = all_edges[eid]['geometry']
                ax.plot(*geom.xy, color='blue', linewidth=3, alpha=0.8, label='Connected (연결됨)')

                adj_nid = e['v'] if e['u'] == self.slit_nid else e['u']
                connected_nodes.add(adj_nid)

        for cut_info in self.main_app.cut_edges_info:
            if cut_info['original_nid'] == self.slit_nid:
                eid = cut_info['edge_id']
                geom = all_edges[eid]['geometry']
                ax.plot(*geom.xy, color='red', linewidth=3, linestyle='--', alpha=0.8, label='Cut (분리됨)')
                dummy_nodes.add(cut_info['dummy_nid'])

        center_pt = self.main_app.graph_nodes[self.slit_nid]
        ax.scatter(center_pt[0], center_pt[1], color='lime', marker='*', s=400, edgecolors='black', zorder=10)
        ax.annotate(f"Slit N{self.slit_nid}\n(Main)", (center_pt[0], center_pt[1]),
                    xytext=(12, 12), textcoords='offset points', fontweight='bold', color='green')

        for nid in connected_nodes:
            if nid == self.slit_nid: continue
            pt = self.main_app.graph_nodes[nid]
            ax.scatter(pt[0], pt[1], color='blue', marker='o', s=100, edgecolors='black', zorder=5)
            ax.annotate(f"N{nid}\n(Adjacent)", (pt[0], pt[1]),
                        xytext=(8, -15), textcoords='offset points', color='blue', fontsize=9)

        for nid in dummy_nodes:
            pt = self.main_app.graph_nodes[nid]
            ax.scatter(pt[0], pt[1], color='red', marker='X', s=150, edgecolors='black', zorder=11)
            ax.annotate(f"N{nid}\n(Dummy)", (pt[0], pt[1]),
                        xytext=(12, -25), textcoords='offset points', color='red', fontweight='bold', fontsize=9)

        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='best')

        self.canvas.draw()


class CellLoopViewerDialog(QDialog):
    def __init__(self, main_app):
        super().__init__(main_app)
        self.main_app = main_app
        self.setWindowTitle("Cell CCW Loop Viewer (반시계 방향 순환 검증)")
        self.resize(800, 800)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "<b>[다중 폐단면 셀(Cell) 반시계 방향 순환 경로 검증]</b><br>"
            "각 방(Cell)을 구성하는 부재들이 <b>반시계 방향(CCW)</b>으로 올바르게 꼬리를 물고 있는지 화살표로 확인합니다."
        )
        layout.addWidget(info_label)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        self.plot_loops()

    def plot_loops(self):
        ax = self.fig.add_subplot(111)

        # 1. 배경: 전체 형상을 옅은 회색으로 렌더링
        for e in self.main_app.graph_edges:
            geom = e['geometry']
            ax.plot(*geom.xy, color='#EEEEEE', linewidth=1, zorder=1)

        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')

        # 2. 실제 곡선 경로(Geometry)를 살려서 렌더링
        for idx, cinfo in enumerate(self.main_app.cells_info):
            color = cmap(idx % 10)
            ordered_nodes = cinfo.get('ordered_nodes', [])
            ordered_edges = cinfo.get('ordered_edges', [])

            if not ordered_edges: continue

            curr_n = ordered_nodes[0]

            for eid in ordered_edges:
                edge_data = self.main_app.graph_edges[eid]
                geom_coords = np.array(edge_data['geometry'].coords)

                # 노드 진행 방향에 맞춰 곡선 좌표계 정렬
                if edge_data['start_node'] == curr_n:
                    path_coords = geom_coords
                    curr_n = edge_data['end_node']
                else:
                    path_coords = geom_coords[::-1]
                    curr_n = edge_data['start_node']

                # 실제 곡선 렌더링
                ax.plot(path_coords[:, 0], path_coords[:, 1], color=color, linewidth=2.5, alpha=0.6, zorder=4)

                # 곡선의 중간 지점(50%)에서 탄젠트 방향으로 화살표 생성
                from shapely.geometry import LineString
                geom_line = LineString(path_coords)

                if geom_line.length > 10.0:
                    pt_mid = geom_line.interpolate(0.5, normalized=True)
                    pt_next = geom_line.interpolate(0.51, normalized=True)  # 살짝 앞쪽 포인트를 잡아 방향 벡터 추출

                    vec = np.array([pt_next.x - pt_mid.x, pt_next.y - pt_mid.y])
                    length = np.linalg.norm(vec)

                    if length > 1e-6:
                        arrow_len = min(geom_line.length * 0.3, 150.0)
                        dv = vec / length * arrow_len

                        ax.annotate('', xy=(pt_mid.x + dv[0], pt_mid.y + dv[1]),
                                    xytext=(pt_mid.x, pt_mid.y),
                                    arrowprops=dict(arrowstyle="->", color=color, lw=3.0, shrinkA=0, shrinkB=0),
                                    zorder=5)

            # 셀 중앙에 번호 라벨 표시
            cx, cy = cinfo['polygon'].centroid.x, cinfo['polygon'].centroid.y
            ax.annotate(f"Cell {cinfo['cell_id']}", (cx, cy), color='white', weight='bold',
                        ha='center', va='center', zorder=10,
                        bbox=dict(boxstyle="circle,pad=0.3", fc=color, ec="none", alpha=0.9))

        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
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

        self.lines_6001 = []
        self.lines_7001 = []
        self.lines_8001 = []
        self.lines_9001 = []

        self.hull_centroid = Point(0, 0)
        self.is_calculated = False

        self.aligned_internal = []
        self.final_healed_centerlines = []

        self.analysis_nodes = []
        self.analysis_elements = []

        self.graph_nodes = {}
        self.graph_edges = []
        self.graph_summary = ""

        self.flowchart_slit_nodes = []

    def init_ui(self):
        self.setStyleSheet("""
            QWidget { font-family: 'Malgun Gothic', 'Noto Sans KR', sans-serif; font-size: 12px; color: #2C3E50; }
            QLineEdit, QComboBox { background-color: #FFFFFF; color: #2C3E50; border: 1px solid #CED4DA; border-radius: 4px; padding: 2px 5px; min-height: 22px; }
            QLineEdit:focus, QComboBox:focus { border: 1.5px solid #00AD1D; }
            QFrame#settingBox { background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 6px; padding: 5px; margin-top: 5px; }
            QLabel { border: none; background: transparent; }
            QPushButton { font-family: 'Malgun Gothic'; font-weight: bold; font-size: 13px; border-radius: 5px; padding: 2px; margin-top: 5px; margin-bottom: 5px; }
            QPushButton#btnGreen { background-color: #00AD1D; color: white; }
            QPushButton#btnGreen:hover { background-color: #009619; }
            QPushButton#btnGreen:disabled { background-color: #A5D6A7; color: #F1F8E9; border: none; }
            QTextEdit#resultBox { font-family: 'Consolas', monospace; font-size: 13px; background-color: #FDFEFE; border: 1px solid #CED4DA; border-radius: 2px; padding: 5px; }
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

        gen_box = QFrame()
        gen_box.setObjectName("settingBox")
        gen_vbox = QVBoxLayout(gen_box)
        gen_vbox.addWidget(QLabel("<b>[General Settings]</b>"))

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

        self.btn_view_loops = QPushButton("3. 셀 순환 경로 확인 🔄")
        self.btn_view_loops.setFixedHeight(40)
        self.btn_view_loops.setObjectName("btnGreen")
        self.btn_view_loops.setEnabled(False)  # 계산 완료 후 활성화
        self.btn_view_loops.clicked.connect(self.show_cell_loops)
        control_panel_layout.addWidget(self.btn_view_loops)

        control_panel_layout.addStretch()
        main_layout.addWidget(control_panel)

        work_area = QWidget()
        work_layout = QVBoxLayout(work_area)
        viz_splitter = QSplitter(Qt.Horizontal)

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

    def group_and_align_centerlines(self, centerlines, tol_dist=150.0, tol_angle=1.5):
        v_lines, h_lines, d_lines = [], [], []

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

        aligned_centerlines = []

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

        d_groups = []
        for info in d_lines:
            p1 = np.array(info['coords'][0])
            ang = info['angle']
            th = np.radians(ang)
            rho = -p1[0] * np.sin(th) + p1[1] * np.cos(th)

            placed = False
            for g in d_groups:
                ang_diff = min(abs(g['avg_angle'] - ang), 180 - abs(g['avg_angle'] - ang))
                if ang_diff < tol_angle and abs(g['avg_rho'] - rho) < tol_dist:
                    g['members'].append(info)
                    tot_len = sum(m['length'] for m in g['members'])
                    g['avg_angle'] = sum(m['angle'] * m['length'] for m in g['members']) / tot_len

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
                d_groups.append({'avg_angle': ang, 'avg_rho': rho, 'members': [info],
                                 'color': colorsys.hsv_to_rgb(np.random.rand(), 0.8, 0.9)})

        for g in d_groups:
            th = np.radians(g['avg_angle'])
            rho = g['avg_rho']

            dir_v = np.array([np.cos(th), np.sin(th)])
            p0 = np.array([-rho * np.sin(th), rho * np.cos(th)])

            for m in g['members']:
                new_coords = []
                for p in m['coords']:
                    pt = np.array(p)
                    t = np.dot(pt - p0, dir_v)
                    proj_pt = p0 + t * dir_v
                    new_coords.append(tuple(proj_pt))

                m['cl']['line'] = LineString(new_coords)
                m['cl']['color'] = g['color']
                aligned_centerlines.append(m['cl'])

        return aligned_centerlines

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
        if not lines: return []
        bridges = []
        groups = {}
        for l in lines:
            c = list(l.coords)
            p1, p2 = np.array(c[0]), np.array(c[-1])
            v = p2 - p1
            L = np.linalg.norm(v)
            if L < 1e-6: continue

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
            segs = sorted([(np.dot(p1, dv), np.dot(p2, dv), p1, p2) for _, p1, p2 in grp],
                          key=lambda x: min(x[0], x[1]))

            for i in range(len(segs) - 1):
                pe = segs[i][3] if segs[i][0] > segs[i][1] else segs[i][2]
                pn = segs[i + 1][2] if segs[i + 1][0] > segs[i + 1][1] else segs[i + 1][3]
                g = np.linalg.norm(pn - pe)

                if 0.1 < g <= threshold_gap:
                    bridges.append(LineString([tuple(pe), tuple(pn)]))
        return lines + bridges

    def robust_heal_1999(self, line_infos, max_gap=500.0):
        if not line_infos: return []
        from shapely.ops import linemerge

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

        endpoints = []
        for i, info in enumerate(new_infos):
            c = list(info['line'].coords)
            if len(c) < 2: continue
            if np.linalg.norm(np.array(c[0]) - np.array(c[-1])) > 1e-3:
                endpoints.append({'idx': i, 'pt': np.array(c[0])})
                endpoints.append({'idx': i, 'pt': np.array(c[-1])})

        used_pts = set()
        bridges = []

        for i, ep1 in enumerate(endpoints):
            if i in used_pts: continue
            best_j = -1
            best_dist = max_gap

            for j, ep2 in enumerate(endpoints):
                if i == j or j in used_pts: continue
                dist = np.linalg.norm(ep1['pt'] - ep2['pt'])
                if dist <= best_dist:
                    best_dist = dist
                    best_j = j

            if best_j != -1:
                ep2 = endpoints[best_j]
                bridges.append({
                    'line': LineString([tuple(ep1['pt']), tuple(ep2['pt'])]),
                    'thickness': 10.0,
                    'type': '1999',
                    'color': '#FF0000'
                })
                used_pts.add(i)
                used_pts.add(best_j)

        return new_infos + bridges

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

            self.raw_1999_lines = [shift(ls) for ls in t_1999]
            m_1999 = unary_union(self.raw_1999_lines)
            self.hull_centroid = m_1999.centroid

            cutters = []
            for c in [shift(ls) for ls in t_1204]:
                c_coords = list(c.coords)
                p1, p2 = np.array(c_coords[0]), np.array(c_coords[-1])
                v = p2 - p1
                L = np.linalg.norm(v)
                if L > 1e-6:
                    u = v / L
                    cutters.append(LineString([tuple(p1 - u * 1000), tuple(p2 + u * 1000)]))

            y_min, y_max = m_1999.bounds[1], m_1999.bounds[3]
            center_cutter = LineString([(0, y_min - 2000), (0, y_max + 2000)])
            cutters.append(center_cutter)

            split_res = split(m_1999, unary_union(cutters)) if cutters else m_1999
            pieces = list(split_res.geoms) if hasattr(split_res, 'geoms') else [split_res]

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

        def filter_short(lines, ml=100.0):
            return [l for l in lines if l.length >= ml]

        def remove_overlapping(lines, dt=10.0, at=5.0):
            lines = sorted(lines, key=lambda x: x.length, reverse=True)
            kept, kept_meta = [], []
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
            segs, cur = [], [coords[0]]
            for i in range(1, len(coords) - 1):
                cur.append(coords[i])
                v1, v2 = np.array(coords[i]) - np.array(coords[i - 1]), np.array(coords[i + 1]) - np.array(coords[i])
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
                meta.append({'ps': ps, 'pe': pe, 'v': v, 'ln': ln, 'unit': v / ln,
                             'ang': np.degrees(np.arctan2(v[1], v[0])) % 180, 'mid': (ps + pe) / 2.0, 'minx': minx,
                             'miny': miny, 'maxx': maxx, 'maxy': maxy})
            used = {i: [] for i in range(len(ls_sorted))}
            pairs = []
            for i in range(len(ls_sorted)):
                if i % 10 == 0:
                    QApplication.processEvents()
                    if progress.wasCanceled(): raise UserWarning("User canceled.")
                if meta[i] is None: continue
                mi = meta[i]
                best_j, best_d, best_ov = -1, float('inf'), None
                mi_ex_minx, mi_ex_maxx = mi['minx'] - max_dist, mi['maxx'] + max_dist
                mi_ex_miny, mi_ex_maxy = mi['miny'] - max_dist, mi['maxy'] + max_dist

                for j in range(i + 1, len(ls_sorted)):
                    if meta[j] is None: continue
                    mj = meta[j]
                    if mj['maxx'] < mi_ex_minx or mj['minx'] > mi_ex_maxx or mj['maxy'] < mi_ex_miny or mj[
                        'miny'] > mi_ex_maxy: continue
                    ad = min(abs(mi['ang'] - mj['ang']), 180 - abs(mi['ang'] - mj['ang']))
                    if ad > angle_tol: continue
                    proj_infinite = mj['ps'] + np.dot(mi['mid'] - mj['ps'], mj['unit']) * mj['unit']
                    d = np.linalg.norm(mi['mid'] - proj_infinite)
                    if d > max_dist: continue
                    t1, t2 = np.dot(mi['ps'] - mj['ps'], mj['unit']), np.dot(mi['pe'] - mj['ps'], mj['unit'])
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
                cs, cl_c = list(short_line.coords), list(long_line.coords)
                ps1, ps2, pl1 = np.array(cs[0]), np.array(cs[-1]), np.array(cl_c[0])
                vl = np.array(cl_c[-1]) - pl1
                ll = np.linalg.norm(vl)
                if ll < 1e-6: continue
                vl_u = vl / ll
                mids = []
                for frac in np.linspace(0, 1, 5):
                    pt_s = ps1 + (ps2 - ps1) * frac
                    t = np.dot(pt_s - pl1, vl_u)
                    mids.append(tuple((pt_s + pl1 + t * vl_u) / 2.0))
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
                ll, dist, shorts = data['long_line'], data['dist'], data['shorts']
                cl_c = list(ll.coords)
                p1, p2 = np.array(cl_c[0]), np.array(cl_c[-1])
                v = p2 - p1
                length = np.linalg.norm(v)
                if length < 1e-6: continue
                vu = v / length

                sl = shorts[0]
                sc = list(sl.coords)
                ps_mid, pl_mid = (np.array(sc[0]) + np.array(sc[-1])) / 2.0, (p1 + p2) / 2.0

                vec = ps_mid - pl_mid
                proj = np.dot(vec, vu) * vu
                perp = vec - proj
                perp_len = np.linalg.norm(perp)

                n = np.array([-vu[1], vu[0]]) if perp_len < 1e-6 else perp / perp_len
                offset = n * (dist / 2.0)
                new_coords = [tuple(np.array(pt) + offset) for pt in cl_c]
                result.append({'line': LineString(new_coords), 'thickness': round(dist * 2) / 2.0})
            return result

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
                        open_ends.append({'line_idx': i, 'end_idx': ei, 'p': p, 'd': d, 'ray': ray})

            updates = {}
            for k, oe in enumerate(open_ends):
                ray, p = oe['ray'], oe['p']
                best_dist, best_p = 10.0 + 1e-5, None

                for j, bg in enumerate(base_geoms):
                    if oe['line_idx'] == j: continue
                    if ray.intersects(bg):
                        inter = ray.intersection(bg)
                        pts = [inter] if inter.geom_type == 'Point' else list(
                            inter.geoms) if inter.geom_type == 'MultiPoint' else []
                        for pt_int in pts:
                            dist = np.linalg.norm(np.array([pt_int.x, pt_int.y]) - p)
                            if 1e-3 < dist < best_dist:
                                best_dist, best_p = dist, (pt_int.x, pt_int.y)

                for j, other_oe in enumerate(open_ends):
                    if k == j or oe['line_idx'] == other_oe['line_idx']: continue
                    other_ray = other_oe['ray']
                    if ray.intersects(other_ray):
                        inter = ray.intersection(other_ray)
                        pts = [inter] if inter.geom_type == 'Point' else list(
                            inter.geoms) if inter.geom_type == 'MultiPoint' else []
                        for pt_int in pts:
                            dist = np.linalg.norm(np.array([pt_int.x, pt_int.y]) - p)
                            if 1e-3 < dist < best_dist:
                                best_dist, best_p = dist, (pt_int.x, pt_int.y)

                if best_p is not None:
                    if oe['line_idx'] not in updates: updates[oe['line_idx']] = {}
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

        def raycast_extend(centerlines, max_dist=300.0):
            extended_pts, result = [], []
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

                    pt_geom = Point(p)
                    conn = False
                    for j, o_geom in enumerate(current_geoms):
                        if i == j: continue
                        if pt_geom.distance(o_geom) <= 1.0:
                            conn = True
                            break
                    if conn: continue

                    ray = LineString([tuple(p), tuple(p + d * max_dist)])
                    rb = ray.bounds
                    bp, bd = None, max_dist

                    for j, o_geom in enumerate(current_geoms):
                        if i == j: continue
                        ob = bounds[j]
                        if rb[2] < ob[0] or rb[0] > ob[2] or rb[3] < ob[1] or rb[1] > ob[3]: continue

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
                                bd, bp = dd, (pt.x, pt.y)

                    if bp:
                        overshoot = 0.1
                        bp_o = (bp[0] + d[0] * overshoot, bp[1] + d[1] * overshoot)
                        if ei == 0:
                            coords[0] = bp_o
                        else:
                            coords[-1] = bp_o
                        extended_pts.append(np.array(bp_o))

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
            unique_cl = []
            seen = set()

            for cl in centerlines:
                c = list(cl['line'].coords)
                if len(c) < 2: continue
                p1_r, p2_r = (round(c[0][0], 3), round(c[0][1], 3)), (round(c[-1][0], 3), round(c[-1][1], 3))

                if np.hypot(p2_r[0] - p1_r[0], p2_r[1] - p1_r[1]) < 0.5: continue
                seg_key = tuple(sorted([p1_r, p2_r]))

                if seg_key not in seen:
                    seen.add(seg_key)
                    unique_cl.append(cl)

            while True:
                endpoints = []
                for cl in unique_cl:
                    c = list(cl['line'].coords)
                    endpoints.extend([(round(c[0][0], 3), round(c[0][1], 3)), (round(c[-1][0], 3), round(c[-1][1], 3))])

                from collections import Counter
                node_degrees = Counter(endpoints)

                to_keep = []
                removed_any = False

                for cl in unique_cl:
                    c = list(cl['line'].coords)
                    p1_r, p2_r = (round(c[0][0], 3), round(c[0][1], 3)), (round(c[-1][0], 3), round(c[-1][1], 3))
                    L = cl['line'].length

                    if (node_degrees[p1_r] == 1 or node_degrees[p2_r] == 1) and L < trim_tol:
                        removed_any = True
                    else:
                        to_keep.append(cl)

                unique_cl = to_keep
                if not removed_any: break

            return unique_cl

        def weld_vertices(centerlines, weld_tol=1.0):
            endpoints = []
            for cl in centerlines:
                c = list(cl['line'].coords)
                endpoints.extend([c[0], c[-1]])

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
                p_s, p_e = welded_nodes.get(c[0], c[0]), welded_nodes.get(c[-1], c[-1])
                if np.hypot(p_s[0] - p_e[0], p_s[1] - p_e[1]) > 0.1:
                    new_item = cl.copy()
                    c[0], c[-1] = p_s, p_e
                    new_item['line'] = LineString(c)
                    new_cl.append(new_item)
            return new_cl

        def heal_collinear_centerlines(centerlines, max_gap=400.0, angle_tol=2.0, align_tol=15.0):
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

            groups = []
            for m in meta:
                placed = False
                for g in groups:
                    g_ref = g[0]
                    ang_diff = min(abs(m['ang'] - g_ref['ang']), 180 - abs(m['ang'] - g_ref['ang']))
                    if ang_diff > angle_tol: continue

                    v_ref, v_target = g_ref['p2'] - g_ref['p1'], g_ref['p1'] - m['p1']
                    d = abs(v_ref[0] * v_target[1] - v_ref[1] * v_target[0]) / g_ref['L']
                    if d <= align_tol:
                        g.append(m)
                        placed = True
                        break
                if not placed:
                    groups.append([m])

            bridges = []
            for g in groups:
                if len(g) < 2: continue
                dv = g[0]['v']
                segs = []
                for m in g:
                    t1, t2 = np.dot(m['p1'], dv), np.dot(m['p2'], dv)
                    if t1 > t2:
                        segs.append({'t_min': t2, 't_max': t1, 'p_min': m['p2'], 'p_max': m['p1'], 'cl': m['cl']})
                    else:
                        segs.append({'t_min': t1, 't_max': t2, 'p_min': m['p1'], 'p_max': m['p2'], 'cl': m['cl']})

                segs.sort(key=lambda x: x['t_min'])

                for i in range(len(segs) - 1):
                    gap = segs[i + 1]['t_min'] - segs[i]['t_max']
                    if 0.1 < gap <= max_gap:
                        new_line = LineString([tuple(segs[i]['p_max']), tuple(segs[i + 1]['p_min'])])
                        thk = (segs[i]['cl'].get('thickness', 10.0) + segs[i + 1]['cl'].get('thickness', 10.0)) / 2.0
                        bridges.append({
                            'line': new_line, 'thickness': thk,
                            'type': segs[i]['cl'].get('type', 'unknown'),
                            'color': segs[i]['cl'].get('color', '#003087')
                        })
            return centerlines + bridges

        try:
            progress.setLabelText("Extracting DXF and Creating Initial 1D Lines...")
            progress.setValue(10)
            QApplication.processEvents()

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

            cl1999 = []
            for ls in l1999s:
                if ls.length > 50.0:
                    cl1999.append({'line': ls, 'thickness': 10.0, 'type': '1999', 'color': '#333333'})

            cl1999 = self.robust_heal_1999(cl1999, max_gap=300.0)

            s1102_raw, s157_raw = [], []
            for l in filter_short(c1102, 10.0): s1102_raw.extend(split_by_slope(l, at=5.0))
            for l in filter_short(c157, 10.0): s157_raw.extend(split_by_slope(l, at=5.0))

            f1102 = remove_overlapping(filter_short(s1102_raw, 50.0), dt=1.0)
            f157 = remove_overlapping(filter_short(s157_raw, 50.0), dt=1.0)

            h1102 = self.heal_internal_collinear(f1102, threshold_gap=150.0)
            h157 = self.heal_internal_collinear(f157, threshold_gap=150.0)

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

            progress.setLabelText("Phase 1: Healing Small Gaps (10mm)...")
            progress.setValue(60)
            all_cl = extend_and_trim_10mm(all_cl)
            all_cl.sort(key=lambda x: 1 if x.get('type') == '157' else 0)

            progress.setLabelText("Phase 2: Short Raycasting (50mm)...")
            progress.setValue(65)
            all_cl, _ = raycast_extend(all_cl, max_dist=50.0)

            progress.setLabelText("Phase 2: Intermediate Topology Cleanup...")
            progress.setValue(75)
            all_cl = split_all_lines_at_intersections(all_cl)
            all_cl = weld_vertices(all_cl, weld_tol=1.0)
            all_cl = clean_topology(all_cl, trim_tol=100.0)

            progress.setLabelText("Phase 3: Deep Raycasting for Decks...")
            progress.setValue(85)
            all_cl, _ = raycast_extend(all_cl, max_dist=400.0)

            progress.setLabelText("Phase 3: Final Topology Cleanup...")
            progress.setValue(90)
            all_cl = split_all_lines_at_intersections(all_cl)
            all_cl = clean_topology(all_cl, trim_tol=300.0)
            all_cl = weld_vertices(all_cl, weld_tol=1.0)

            progress.setLabelText("Phase 4: Topology Cleanup...")
            progress.setValue(90)
            all_cl = split_all_lines_at_intersections(all_cl)

            progress.setLabelText("Phase 5: Welding Vertices...")
            progress.setValue(98)
            all_cl = weld_vertices(all_cl, weld_tol=1.0)

            progress.setLabelText("Phase 6: Removing Duplicates and Trimming...")
            progress.setValue(95)
            all_cl = clean_topology(all_cl, trim_tol=100.0)

            self.hull_only_centerlines = [cl.copy() for cl in all_cl]

            # ✨ [요구사항 1, 6 반영] 6001~9001 보강재 중심선 매칭 및 생성
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
                # 요구사항 6: 6001~9001 스티프너는 연한 회색으로 시각화
                c['color'] = '#D3D3D3'

            all_cl.extend(stiff_cl)
            self.final_healed_centerlines = all_cl

            progress.setLabelText("Building Mathematical Graph...")
            self.build_graph()

            progress.setLabelText("Detecting Closed Cells (Loops)...")
            self.detect_closed_cells()

            progress.setLabelText("Executing Flowchart Algorithm...")
            self.execute_flowchart_algorithm()

            try:
                Vy_input_t = float(self.txt_vy.text().strip())
                Vy_input_N = Vy_input_t * 9806.65
            except ValueError:
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

            # 순수 I_xx 계산 시에도 잘려나간 우현을 무시하고, 전단류 계산과 완벽히 동일한 반단면 박스로 잘라서 통일
            keep_box = box(-9999999.0, -9999999.0, self.x_cut + 0.5, 9999999.0)

            for cl in all_cl:
                geom = cl['line']
                clipped = geom.intersection(keep_box)
                if clipped.is_empty: continue
                geoms = [clipped] if clipped.geom_type == 'LineString' else \
                    list(clipped.geoms) if clipped.geom_type == 'MultiLineString' else []

                thk = cl.get('thickness', 10.0)
                if thk <= 0: thk = 10.0

                for g in geoms:
                    coords = list(g.coords)
                    for i in range(len(coords) - 1):
                        x1, y1 = coords[i]
                        x2, y2 = coords[i + 1]
                        dx, dy = x2 - x1, y2 - y1
                        L = np.hypot(dx, dy)
                        if L < 1e-6: continue

                        a = L * thk
                        yc = (y1 + y2) / 2.0

                        total_area += a
                        sum_qx += a * yc
                        segments_1d.append((a, yc, dx, dy, L))

            if total_area > 0:
                na_y = sum_qx / total_area
                ixx = 0.0
                for a, yc, dx, dy, L in segments_1d:
                    i_local = (a * (dy ** 2)) / 12.0
                    ixx += i_local + a * ((yc - na_y) ** 2)

                self.pure_hull_ixx_half = ixx
                self.pure_hull_na_y = na_y
                self.calc_total_area = total_area
                self.calc_na_bl = na_y

                calc_result_text = (
                    f"\n\n📊 [Pure Hull Properties (Half-Section Base)]\n"
                    f"----------------------------------------\n"
                    f"- Total Area (반단면적)  : {total_area:,.2f} mm²\n"
                    f"- N.A from Base (중립축) : {na_y:,.2f} mm\n"
                    f"- Half Moment of Inertia : {ixx:,.0f} mm⁴\n"
                    f"- Full Moment of Inertia : {ixx * 2:,.0f} mm⁴\n"
                    f"----------------------------------------\n"
                )
            else:
                calc_result_text = "\n\n❌ 유효한 단면적이 없어 이너시아를 계산할 수 없습니다."

            shear_ixx_text = ""
            if hasattr(self, 'shear_calc_ixx_half'):
                shear_ixx_text = (
                    f"\n\n🟦 [Shear Flow Calc Properties (Hull + Mass Points)]\n"
                    f"- ✅ 귀속된 질점 총합 면적: {getattr(self, 'total_stiffener_area', 0.0):,.2f} mm²\n"
                    f"- Hull + Mass Point 반단면적: {getattr(self, 'shear_calc_area_half', 0.0):,.2f} mm²\n"
                    f"- Hull + Mass Point 중립축 : {self.shear_calc_na_y:,.2f} mm\n"
                    f"- Half Moment of Inertia   : {self.shear_calc_ixx_half:,.0f} mm⁴\n"
                    f"- Full Moment of Inertia   : {self.shear_calc_ixx_full:,.0f} mm⁴\n"
                    f"(※ Point Mass 변환으로 인해 I_local 값 분실로 극소량의 차이가 발생할 수 있으나 정상입니다.)"
                )

            summary_text = (
                "✅ 1D Transformation & Full Healing Complete!\n\n"
                f"- Extracted Outer Hull Lines (1999): {len(cl1999)}\n"
                f"- Aligned Internal Lines (1102, 157): {len(self.aligned_internal)}\n"
                f"- Added Stiffeners (6001~9001): {len(stiff_cl)}\n"
                f"- Final Healed Elements: {len(all_cl)}"
                f"{calc_result_text}"
                f"{self.graph_summary}"
                f"{getattr(self, 'cell_summary', '')}"
                f"{shear_ixx_text}"
            )
            self.result_box.setText(summary_text)

            self.is_calculated = True
            progress.setValue(100)
            self.refresh_ui()

        except Exception as e:
            self.result_box.setText(f"❌ Error:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            progress.close()
            self.is_processing = False
            self.btn_calc.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.btn_view_loops.setEnabled(True)

    def build_graph(self):
        from collections import defaultdict

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
        keep_box = box(-9999999.0, -9999999.0, self.x_cut + 0.5, 9999999.0)

        raw_nodes = {}
        raw_edges = []
        node_map = {}
        nid_counter = 1

        def get_nid(pt):
            nonlocal nid_counter
            for ept, eid in node_map.items():
                if np.hypot(ept[0] - pt[0], ept[1] - pt[1]) < 1.0:
                    return eid
            node_map[pt] = nid_counter
            raw_nodes[nid_counter] = pt
            nid_counter += 1
            return node_map[pt]

        for cl in self.final_healed_centerlines:
            if cl.get('type') in ['6001', '7001', '8001', '9001', 'stiffener']:
                continue

            geom = cl['line']
            clipped = geom.intersection(keep_box)
            if clipped.is_empty: continue

            geoms = [clipped] if clipped.geom_type == 'LineString' else \
                list(clipped.geoms) if clipped.geom_type == 'MultiLineString' else []

            for g in geoms:
                if g.length < 1.0: continue
                c = list(g.coords)
                u, v = get_nid(c[0]), get_nid(c[-1])
                raw_edges.append({
                    'u': u, 'v': v, 'coords': c,
                    'thickness': cl.get('thickness', 10.0),
                    'type': cl.get('type', 'unknown')
                })

        while True:
            node_to_edges = defaultdict(list)
            for i, e in enumerate(raw_edges):
                if e is None: continue
                node_to_edges[e['u']].append(i)
                if e['u'] != e['v']: node_to_edges[e['v']].append(i)

            merged_any = False
            for nid, edge_indices in node_to_edges.items():
                if len(edge_indices) == 2:
                    node_pt = raw_nodes[nid]
                    if abs(node_pt[0] - self.x_cut) <= 1.5: continue

                    e1_idx, e2_idx = edge_indices[0], edge_indices[1]
                    if e1_idx == e2_idx: continue
                    e1, e2 = raw_edges[e1_idx], raw_edges[e2_idx]

                    if abs(e1['thickness'] - e2['thickness']) > 0.1 or e1['type'] != e2['type']: continue

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
                        'u': u_new, 'v': v_new, 'coords': merged_coords,
                        'thickness': e1['thickness'], 'type': e1['type']
                    }
                    raw_edges[e2_idx] = None
                    merged_any = True
                    break

            if not merged_any: break

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
                'id': i, 'start_node': e['u'], 'start_coord': self.graph_nodes[e['u']],
                'end_node': e['v'], 'end_coord': self.graph_nodes[e['v']],
                'thickness': e['thickness'], 'length': geom.length,
                'geometry': geom, 'type': e['type'],
                'mass_points': []  # 초기화
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
        from shapely.ops import polygonize, unary_union
        from shapely.geometry import LinearRing
        from collections import defaultdict
        import numpy as np

        self.cells_info = []
        self.edge_to_cells = {e['id']: [] for e in self.graph_edges}

        lines = [e['geometry'] for e in self.graph_edges]
        merged_lines = unary_union(lines)
        polygons = list(polygonize(merged_lines))

        for i, poly in enumerate(polygons):
            cell_id = i + 1
            cell_area = poly.area
            cell_edges = []

            bound = poly.boundary
            for eid, e in enumerate(self.graph_edges):
                geom = e['geometry']
                # 곡선의 오차 방지를 위해 정중앙점(interpolate 0.5)을 사용해 엣지 포함 여부 엄격 판별
                pt_on_line = geom.interpolate(0.5, normalized=True)
                if bound.distance(pt_on_line) < 1e-2:
                    cell_edges.append(eid)
                    self.edge_to_cells[eid].append(cell_id)

            self.cells_info.append({
                'cell_id': cell_id,
                'polygon': poly,
                'area': cell_area,
                'edges': cell_edges
            })

        # ✨ 각 셀의 실제 곡선 경로 전체 좌표계를 조립하여 반시계(CCW) 판별
        for cinfo in self.cells_info:
            cell_edges = cinfo['edges']
            if not cell_edges: continue

            adj = defaultdict(list)
            for eid in cell_edges:
                e = self.graph_edges[eid]
                adj[e['start_node']].append((e['end_node'], eid))
                adj[e['end_node']].append((e['start_node'], eid))

            start_node = list(adj.keys())[0]
            curr_node = start_node
            ordered_nodes = [curr_node]
            ordered_edges = []
            visited_edges = set()

            # 한붓그리기 경로 추적
            while len(visited_edges) < len(cell_edges):
                moved = False
                for nxt, eid in adj[curr_node]:
                    if eid not in visited_edges:
                        visited_edges.add(eid)
                        ordered_edges.append(eid)
                        curr_node = nxt
                        ordered_nodes.append(curr_node)
                        moved = True
                        break
                if not moved: break

            # ✨ [핵심 교체] 직선 노드가 아닌 곡선 전체를 하나의 링(Ring)으로 조립
            full_path_coords = []
            curr_n = start_node

            for eid in ordered_edges:
                edge_data = self.graph_edges[eid]
                geom_coords = list(edge_data['geometry'].coords)

                if edge_data['start_node'] == curr_n:
                    full_path_coords.extend(geom_coords[:-1])
                    curr_n = edge_data['end_node']
                else:
                    full_path_coords.extend(geom_coords[::-1][:-1])
                    curr_n = edge_data['start_node']

            # 루프 닫기
            full_path_coords.append(full_path_coords[0])

            # Shapely의 내부 연산을 통해 완벽하게 정밀한 곡선 기반 방향성 도출
            ring = LinearRing(full_path_coords)

            # 만약 시계방향(CW)이라면 배열의 순서를 완전히 뒤집어 반시계(CCW)로 교정
            if not ring.is_ccw:
                ordered_nodes.reverse()
                ordered_edges.reverse()

            cinfo['ordered_nodes'] = ordered_nodes
            cinfo['ordered_edges'] = ordered_edges

    def execute_flowchart_algorithm(self):
        sim_edges = {e['id']: {'u': e['start_node'], 'v': e['end_node']} for e in self.graph_edges}
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

                    del sim_edges[m]

                    if deg_j == 1 or deg_j >= 3:
                        break
                    else:
                        i = j

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
                    slit_nodes_ids.append(j)

                    new_nid = max(self.graph_nodes.keys()) + 1
                    self.graph_nodes[new_nid] = self.graph_nodes[j]

                    if working_edges[m]['u'] == j:
                        working_edges[m]['u'] = new_nid
                    else:
                        working_edges[m]['v'] = new_nid

                    self.cut_edges_info.append({'edge_id': m, 'original_nid': j, 'dummy_nid': new_nid})

                    del sim_edges[m]
                    break
                else:
                    visited_nodes.add(j)
                    i = j

        self.flowchart_slit_nodes = list(set(slit_nodes_ids))
        self.remaining_edges_info = working_edges

    # ✨ [요구사항 2, 3, 4 처리] 보강재 질점 매핑 함수 (무게중심 좌표 추가)
    def process_stiffener_mass_points(self):
        keep_box = box(-9999999.0, -9999999.0, self.x_cut + 0.5, 9999999.0)
        stiff_lines = []

        # 반단면 컷팅된 스티프너만 수집
        for cl in self.final_healed_centerlines:
            if cl.get('type') == 'stiffener':
                clipped = cl['line'].intersection(keep_box)
                if clipped.is_empty: continue
                geoms = [clipped] if clipped.geom_type == 'LineString' else \
                    list(clipped.geoms) if clipped.geom_type == 'MultiLineString' else []
                for g in geoms:
                    if g.length > 1.0:
                        stiff_lines.append({'line': g, 'thickness': cl.get('thickness', 10.0)})

        groups = []
        used = set()
        for i, s1 in enumerate(stiff_lines):
            if i in used: continue
            curr_group = [s1]
            used.add(i)

            q = [s1['line']]
            while q:
                curr_geom = q.pop(0)
                for j, s2 in enumerate(stiff_lines):
                    if j not in used:
                        if curr_geom.distance(s2['line']) < 1.0:
                            used.add(j)
                            curr_group.append(s2)
                            q.append(s2['line'])
            groups.append(curr_group)

        for edge in self.graph_edges:
            edge['mass_points'] = []

        # 가장 가까운 외판/격벽에 스티프너 무조건 귀속
        for grp in groups:
            grp_area = 0.0
            grp_qy = 0.0

            for s in grp:
                thk = s.get('thickness', 10.0)
                if thk < 1.0: thk = 10.0  # 스티프너 두께 0 수렴 방지
                area = s['line'].length * thk
                grp_area += area
                grp_qy += area * s['line'].centroid.y

            centroid_y = grp_qy / grp_area if grp_area > 0 else 0.0
            grp_geom = unary_union([s['line'] for s in grp])  # 스티프너 전체 형상

            best_eid = -1
            min_dist = float('inf')

            for edge in self.graph_edges:
                if edge['id'] not in self.remaining_edges_info: continue

                dist = grp_geom.distance(edge['geometry'])
                if dist < min_dist:
                    min_dist = dist
                    best_eid = edge['id']

            # 500mm 내의 가장 가까운 엣지(157, -1102, 1999)에 100% 매칭
            if best_eid != -1 and min_dist < 500.0:
                p_edge, _ = nearest_points(self.graph_edges[best_eid]['geometry'], grp_geom)
                self.graph_edges[best_eid]['mass_points'].append({
                    'pt': (p_edge.x, p_edge.y),
                    'centroid_y': centroid_y,
                    'area': grp_area
                })

    def calculate_determinate_shear_flow(self, Vy_total=1000000.0):
        # ✨ 스티프너를 질점(Mass Point)으로 변환 후 귀속
        self.process_stiffener_mass_points()

        nodes = self.graph_nodes
        edges = {}
        for eid, e_info in self.remaining_edges_info.items():
            if self.graph_edges[eid].get('type') != 'stiffener':
                edges[eid] = e_info

        # 1. 질점을 포함한 단면적 및 중립축(NA_y) 계산
        total_area = 0.0
        sum_qx = 0.0
        for eid, e in edges.items():
            geom = self.graph_edges[eid]['geometry']
            thk = self.graph_edges[eid].get('thickness', 10.0)
            L = geom.length
            if L < 1e-6: continue
            a = L * thk
            yc = geom.centroid.y
            total_area += a
            sum_qx += a * yc

            for mp in self.graph_edges[eid].get('mass_points', []):
                total_area += mp['area']
                sum_qx += mp['area'] * mp['centroid_y']

        if total_area < 1e-6: return
        na_y = sum_qx / total_area
        self.shear_calc_area_half = total_area  # 추적용 변수 추가

        # 2. 질점을 포함한 반단면 기준 이너시아(Ixx_half) 계산
        ixx_half = 0.0
        for eid, e in edges.items():
            geom = self.graph_edges[eid]['geometry']
            thk = self.graph_edges[eid].get('thickness', 10.0)
            num_chunks = max(1, int(geom.length / 50.0))
            chunk_len = geom.length / num_chunks
            for i in range(num_chunks):
                sub = substring(geom, i * chunk_len, (i + 1) * chunk_len)
                a = sub.length * thk
                yc = sub.centroid.y
                c = list(sub.coords)
                dy = c[-1][1] - c[0][1]
                i_local = a * (dy ** 2) / 12.0
                ixx_half += i_local + a * (yc - na_y) ** 2

            for mp in self.graph_edges[eid].get('mass_points', []):
                ixx_half += mp['area'] * (mp['centroid_y'] - na_y) ** 2

        if ixx_half == 0: return

        self.shear_calc_ixx_half = ixx_half
        self.shear_calc_ixx_full = ixx_half * 2
        self.shear_calc_na_y = na_y

        # 전체 부착된 스티프너 면적 검증용
        self.total_stiffener_area = sum(mp['area'] for e in self.graph_edges for mp in e.get('mass_points', []))

        from collections import defaultdict
        nm = defaultdict(int)
        for e in edges.values():
            nm[e['u']] += 1
            nm[e['v']] += 1

        vn = defaultdict(int)
        vm = {eid: 0 for eid in edges}

        node_flows_unit = defaultdict(dict)
        self.edge_q_results = {}
        self.calc_route = []

        while True:
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

                        geom = self.graph_edges[m]['geometry']
                        thk = self.graph_edges[m].get('thickness', 10.0)
                        coord_u = nodes[i_curr]
                        geom_start, geom_end = geom.coords[0], geom.coords[-1]

                        dist_to_start = np.hypot(coord_u[0] - geom_start[0], coord_u[1] - geom_start[1])
                        dist_to_end = np.hypot(coord_u[0] - geom_end[0], coord_u[1] - geom_end[1])
                        is_forward = dist_to_start < dist_to_end

                        self.calc_route.append({'phase': 1, 'edge_id': m, 'from_node': i_curr, 'to_node': j_next,
                                                'is_forward': is_forward})

                        L_total = geom.length
                        num_chunks = max(1, int(np.ceil(L_total / 50.0)))
                        sub_results = []
                        unprocessed_mps = self.graph_edges[m].get('mass_points', []).copy()

                        for k in range(num_chunks):
                            if is_forward:
                                d1, d2 = k * (L_total / num_chunks), (k + 1) * (L_total / num_chunks)
                            else:
                                d1, d2 = L_total - (k + 1) * (L_total / num_chunks), L_total - k * (
                                            L_total / num_chunks)

                            sub_geom = substring(geom, min(d1, d2), max(d1, d2))
                            if sub_geom.length < 1e-6: continue
                            A_sub = sub_geom.length * thk
                            y_bar = sub_geom.centroid.y - na_y
                            dS_z = A_sub * y_bar

                            # ✨ 해당 구간에 질점(스티프너)이 포함되어 있는지 확인 및 누적 모멘트 dS_z 추가 (실제 무게중심 적용)
                            for mp in list(unprocessed_mps):
                                mp_dist = geom.project(Point(mp['pt']))
                                if min(d1, d2) - 1e-3 <= mp_dist <= max(d1, d2) + 1e-3:
                                    dS_z += mp['area'] * (mp['centroid_y'] - na_y)
                                    unprocessed_mps.remove(mp)

                            if k == num_chunks - 1:
                                for mp in unprocessed_mps:
                                    dS_z += mp['area'] * (mp['centroid_y'] - na_y)
                                unprocessed_mps.clear()

                            dq_unit = - (0.5 / ixx_half) * dS_z
                            q_next_unit = q_curr_unit + dq_unit

                            sub_results.append(
                                {'geom': sub_geom, 'q_start_unit': q_curr_unit, 'q_end_unit': q_next_unit,
                                 'is_forward': is_forward})
                            q_curr_unit = q_next_unit

                        self.edge_q_results[m] = sub_results
                        node_flows_unit[j_next][m] = q_curr_unit

                        vm[m] += 1
                        vn[j_next] += 1
                        i_curr = j_next

                        if nm[j_next] == 1 or nm[j_next] >= 3: break

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
                                geom, thk = self.graph_edges[m]['geometry'], self.graph_edges[m].get('thickness', 10.0)
                                coord_u = nodes[i_curr]
                                geom_start, geom_end = geom.coords[0], geom.coords[-1]

                                dist_to_start = np.hypot(coord_u[0] - geom_start[0], coord_u[1] - geom_start[1])
                                dist_to_end = np.hypot(coord_u[0] - geom_end[0], coord_u[1] - geom_end[1])
                                is_forward = dist_to_start < dist_to_end

                                self.calc_route.append(
                                    {'phase': 2, 'edge_id': m, 'from_node': i_curr, 'to_node': j_next,
                                     'is_forward': is_forward})

                                L_total = geom.length
                                num_chunks = max(1, int(np.ceil(L_total / 50.0)))
                                sub_results = []
                                unprocessed_mps = self.graph_edges[m].get('mass_points', []).copy()

                                for k in range(num_chunks):
                                    if is_forward:
                                        d1, d2 = k * (L_total / num_chunks), (k + 1) * (L_total / num_chunks)
                                    else:
                                        d1, d2 = L_total - (k + 1) * (L_total / num_chunks), L_total - k * (
                                                    L_total / num_chunks)

                                    sub_geom = substring(geom, min(d1, d2), max(d1, d2))
                                    if sub_geom.length < 1e-6: continue
                                    A_sub = sub_geom.length * thk
                                    y_bar = sub_geom.centroid.y - na_y
                                    dS_z = A_sub * y_bar

                                    # ✨ 해당 구간에 질점(스티프너)이 포함되어 있는지 확인 및 누적 모멘트 dS_z 추가 (실제 무게중심 적용)
                                    for mp in list(unprocessed_mps):
                                        mp_dist = geom.project(Point(mp['pt']))
                                        if min(d1, d2) - 1e-3 <= mp_dist <= max(d1, d2) + 1e-3:
                                            dS_z += mp['area'] * (mp['centroid_y'] - na_y)
                                            unprocessed_mps.remove(mp)

                                    if k == num_chunks - 1:
                                        for mp in unprocessed_mps:
                                            dS_z += mp['area'] * (mp['centroid_y'] - na_y)
                                        unprocessed_mps.clear()

                                    dq_unit = - (0.5 / ixx_half) * dS_z
                                    q_next_unit = q_curr_unit + dq_unit

                                    sub_results.append(
                                        {'geom': sub_geom, 'q_start_unit': q_curr_unit, 'q_end_unit': q_next_unit,
                                         'is_forward': is_forward})
                                    q_curr_unit = q_next_unit

                                self.edge_q_results[m] = sub_results
                                node_flows_unit[j_next][m] = q_curr_unit

                                vm[m] += 1
                                vn[j_next] += 1
                                i_curr = j_next

                                if nm[j_next] == 1 or nm[j_next] >= 3: break

                if not bridge_started: break

        self.user_Vy_total = Vy_total

    def on_edge_click(self, event):
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

                if 'mass_points' in edge and edge['mass_points']:
                    info_text += f" - 부착된 보강재 질점 개수: {len(edge['mass_points'])} 개\n"
                    tot_mass_area = sum([m['area'] for m in edge['mass_points']])
                    info_text += f" - 부착된 스티프너 총면적: {tot_mass_area:,.2f} mm²\n"

                if hasattr(self, 'edge_q_results') and edge_id in self.edge_q_results:
                    chunks = self.edge_q_results[edge_id]
                    q_s_unit, q_e_unit = chunks[0]['q_start_unit'], chunks[-1]['q_end_unit']

                    user_vy = getattr(self, 'user_Vy_total', 1000000.0)
                    q_s_actual, q_e_actual = q_s_unit * user_vy, q_e_unit * user_vy

                    thk = edge['thickness']
                    tau_s_actual, tau_e_actual = q_s_actual / thk, q_e_actual / thk

                    if hasattr(self, 'edge_to_cells') and edge_id in self.edge_to_cells:
                        belonging_cells = self.edge_to_cells[edge_id]
                        if len(belonging_cells) == 0:
                            shared_text = "일반 개단면 부재 (폐루프 미포함)"
                        elif len(belonging_cells) == 1:
                            shared_text = f"외판/독립 격벽 (Cell {belonging_cells[0]} 단독 소속)"
                        else:
                            shared_text = f"🔥 공유 격벽 (Shared Web) - Cell {belonging_cells} 사이 연결"

                        info_text += f"\n\n🟩 [위상수학적 셀(Cell) 정보]\n - 소속 상태 : {shared_text}\n"

                QMessageBox.information(self, f"Edge ID: {edge_id}", info_text)

    def on_slit_node_click(self, event):
        if event.artist and hasattr(event.artist, 'get_gid'):
            gid = event.artist.get_gid()
            if gid is not None and str(gid).startswith("slit_"):
                nid = int(str(gid).split("_")[1])
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
            # [도면 1, 2] 배경 스티프너 및 외판 렌더링
            if hasattr(self, 'final_healed_centerlines'):
                # 도면 2 반단면 컷팅을 위한 박스 생성
                keep_box = box(-9999999.0, -9999999.0, getattr(self, 'x_cut', 0.0) + 0.5, 9999999.0)

                for cl in self.final_healed_centerlines:
                    lo = cl['line']
                    c_type = cl.get('type')
                    final_color = '#000000' if c_type == '1999' else ('#D3D3D3' if c_type == 'stiffener' else '#003087')
                    thk = cl.get('thickness', 10.0)
                    visual_thk = thk if thk >= 50.0 else 50.0

                    # 1번 도면(전체 형상) 렌더링
                    if visual_thk > 0:
                        try:
                            poly = lo.buffer(visual_thk / 2.0, cap_style=2)
                            if poly.geom_type == 'Polygon':
                                ax1.fill(*poly.exterior.xy, color=final_color, alpha=0.3, zorder=9, edgecolor='none')
                            elif poly.geom_type == 'MultiPolygon':
                                for p in poly.geoms: ax1.fill(*p.exterior.xy, color=final_color, alpha=0.3, zorder=9,
                                                              edgecolor='none')
                        except:
                            pass
                    ax1.plot(*lo.xy, color=final_color, linewidth=2.0, alpha=0.9, zorder=10)

                    # ✨ 2번 도면(전단류 뷰) 렌더링 - 스티프너 반단면 컷팅 적용
                    if c_type == 'stiffener':
                        clipped_lo = lo.intersection(keep_box)
                        if not clipped_lo.is_empty:
                            c_geoms = [clipped_lo] if clipped_lo.geom_type == 'LineString' else \
                                list(clipped_lo.geoms) if clipped_lo.geom_type == 'MultiLineString' else []
                            for cg in c_geoms:
                                ax2.plot(*cg.xy, color='#D3D3D3', linewidth=2.0, alpha=0.9, zorder=4)

            if hasattr(self, 'graph_edges'):
                if hasattr(self, 'cells_info') and self.cells_info:
                    for cinfo in self.cells_info:
                        poly, cid = cinfo['polygon'], cinfo['cell_id']
                        ax2.annotate(f"Cell {cid}", (poly.centroid.x, poly.centroid.y), color='black', weight='bold',
                                     fontsize=12,
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
                            tau_start, tau_end = chunk['q_start_unit'] * user_vy / thk, chunk[
                                'q_end_unit'] * user_vy / thk
                            max_tau = max(max_tau, abs(tau_start), abs(tau_end))

                    nodes_arr = np.array(list(self.graph_nodes.values()))
                    max_dim = max(nodes_arr[:, 0].max() - nodes_arr[:, 0].min(),
                                  nodes_arr[:, 1].max() - nodes_arr[:, 1].min()) if len(nodes_arr) > 0 else 10000.0

                    visual_scale = (max_dim * 0.05) / max_tau if max_tau > 1e-9 else 1.0
                    cmap = cm.turbo
                    norm = mcolors.Normalize(vmin=0, vmax=max_tau)

                    for eid, edge_data in enumerate(self.graph_edges):
                        res_list = self.edge_q_results.get(eid)
                        thk = edge_data.get('thickness', 10.0)
                        if thk < 0.1: thk = 0.1

                        if res_list:
                            geom_c = list(edge_data['geometry'].coords)
                            dx, dy = geom_c[-1][0] - geom_c[0][0], geom_c[-1][1] - geom_c[0][1]
                            base_tg = np.array([dx, dy])
                            base_tg = base_tg / np.linalg.norm(base_tg) if np.linalg.norm(base_tg) > 1e-6 else np.array(
                                [1.0, 0.0])

                            is_fwd = res_list[0].get('is_forward', True)
                            chunks_ordered = res_list if is_fwd else list(reversed(res_list))

                            all_c_pts, all_top_pts, all_taus = [], [], []

                            for chunk in chunks_ordered:
                                sub_geom = chunk['geom']
                                tau_s_signed = (chunk['q_start_unit'] * user_vy) / thk
                                tau_e_signed = (chunk['q_end_unit'] * user_vy) / thk
                                c = np.array(sub_geom.coords)
                                num_pts = len(c)

                                taus_signed = np.linspace(tau_s_signed, tau_e_signed,
                                                          num_pts) if is_fwd else np.linspace(tau_e_signed,
                                                                                              tau_s_signed, num_pts)
                                n_vecs = np.zeros_like(c)

                                for p_idx in range(num_pts):
                                    if num_pts == 2 or p_idx == 0:
                                        tg = c[1] - c[0]
                                    elif p_idx == num_pts - 1:
                                        tg = c[-1] - c[-2]
                                    else:
                                        tg = c[p_idx + 1] - c[p_idx - 1]

                                    tg_len = np.linalg.norm(tg)
                                    tg = tg / tg_len if tg_len > 1e-6 else base_tg
                                    n_vecs[p_idx] = np.array([-tg[1], tg[0]])

                                top_pts = c + n_vecs * (taus_signed[:, np.newaxis] * visual_scale)

                                if len(all_c_pts) > 0:
                                    all_c_pts.extend(c[1:]), all_top_pts.extend(top_pts[1:]), all_taus.extend(
                                        taus_signed[1:])
                                else:
                                    all_c_pts.extend(c), all_top_pts.extend(top_pts), all_taus.extend(taus_signed)

                            all_c_pts, all_top_pts, all_taus = np.array(all_c_pts), np.array(all_top_pts), np.array(
                                all_taus)
                            for idx_pt in range(len(all_c_pts) - 1):
                                px = [all_c_pts[idx_pt, 0], all_c_pts[idx_pt + 1, 0], all_top_pts[idx_pt + 1, 0],
                                      all_top_pts[idx_pt, 0]]
                                py = [all_c_pts[idx_pt, 1], all_c_pts[idx_pt + 1, 1], all_top_pts[idx_pt + 1, 1],
                                      all_top_pts[idx_pt, 1]]
                                fc = cmap(norm(np.mean(np.abs(all_taus[idx_pt:idx_pt + 2]))))

                                ax2.fill(px, py, color=fc, alpha=0.6, edgecolor='none', zorder=8)
                                ax2.plot([all_top_pts[idx_pt, 0], all_top_pts[idx_pt + 1, 0]],
                                         [all_top_pts[idx_pt, 1], all_top_pts[idx_pt + 1, 1]], color=fc, linewidth=1.5,
                                         zorder=9)

                            geom = edge_data['geometry']
                            ax2.plot(*geom.xy, color='black', linewidth=1.5, zorder=10, picker=5, gid=f"edge_{eid}")
                        else:
                            geom = edge_data['geometry']
                            ax2.plot(*geom.xy, color='dodgerblue', linewidth=2.5, alpha=0.8, zorder=10, picker=5,
                                     gid=f"edge_{eid}")

                        if 'mass_points' in edge_data:
                            for mp in edge_data['mass_points']:
                                ax2.scatter(mp['pt'][0], mp['pt'][1], color='darkgray', s=35, marker='o', zorder=35,
                                            edgecolors='black')

                    cut_edge_ids = [c['edge_id'] for c in self.cut_edges_info] if hasattr(self,
                                                                                          'cut_edges_info') else []
                    for eid in cut_edge_ids:
                        if eid < len(self.graph_edges):
                            geom = self.graph_edges[eid]['geometry']
                            ax2.plot(*geom.xy, color='#BDC3C7', linewidth=2.5, linestyle='--', zorder=5)

                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    self.cbar = self.fig2.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
                    self.cbar.set_label(f'Signed Shear Stress τ_d (MPa)\n[Total Vy = {user_vy:,.0f} N 적용]',
                                        fontweight='bold', fontsize=10)

                node_xs, node_ys = [pt[0] for pt in self.graph_nodes.values()], [pt[1] for pt in
                                                                                 self.graph_nodes.values()]
                ax2.scatter(node_xs, node_ys, color='red', s=40, zorder=20, edgecolors='black')

                for nid, pt in self.graph_nodes.items():
                    ax2.annotate(str(nid), (pt[0], pt[1]), xytext=(4, 4), textcoords='offset points', color='black',
                                 fontsize=9, fontweight='bold', zorder=25)

                if hasattr(self, 'flowchart_slit_nodes') and self.flowchart_slit_nodes:
                    for i, nid in enumerate(self.flowchart_slit_nodes):
                        if nid in self.graph_nodes:
                            pt, label = self.graph_nodes[nid], 'Slit Position (Cut Node)' if i == 0 else ""
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

    def show_cell_loops(self):
        """셀 순환 검증 팝업창을 띄웁니다."""
        dialog = CellLoopViewerDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UltimateShipAnalyzer()
    win.show()
    sys.exit(app.exec())
    

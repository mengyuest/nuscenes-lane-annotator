# Import necessary modules
import os
import sys
import time
import argparse
import numpy as np
import pickle
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QRect
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, \
    QCheckBox, QLabel, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,\
    QSlider, QListView, QTableView, QSizePolicy, QGraphicsPixmapItem, QFrame, QTextEdit, QRadioButton,\
    QButtonGroup, QTabWidget, QTableWidget, QTableWidgetItem, QComboBox, QAbstractItemView,\
    QMessageBox

from PyQt5.QtGui import QIcon,QPainter, QBrush, QColor, QPixmap, QImage, QStandardItemModel,\
    QStandardItem, QPen

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from utils import MyTF, MyPainter, remove_qimage_margin, get_lanes_nearby, visualize_nuscenes_scene, visualize_nuscenes_legends

class CanvasWidget(QGraphicsView):
    photoClicked = pyqtSignal(QPointF)
    doubleClicked = pyqtSignal(QPointF)
    def __init__(self):
        super().__init__()
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0
    
    def setPhoto(self, pixmap=None, dont_fit_view=False):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        if not dont_fit_view:
            self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def getPixMapCoord(self, pos):
        return self.mapToScene(pos)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.getPixMapCoord(event.pos()).toPoint())
        super(CanvasWidget, self).mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):        
        if self._photo.isUnderMouse():
            self.doubleClicked.emit(self.getPixMapCoord(event.pos()).toPoint())
        # Call the base class implementation
        super(CanvasWidget, self).mouseDoubleClickEvent(event)

class MyGUIApp:
    def __init__(self):
        # Create the application instance
        self.setup_ui()    
        
        # model variables
        # cache related
        self.cache = {}
        self.annotated_data = {0:{"high_level": None, "lanes":{"curr":[], "left":[], "right":[]}}}
        self.qimage_cache = None
        
        self.is_loaded = False
        self.ctrl_pressed = False
        self.cur_ti = None        
        self.hover_x = None
        self.hover_y = None
        self.ego_x_pixel = None
        self.ego_y_pixel = None
        self.plot_lanes = None
        self.highlighted_lane = None
        self.highlighted_tracked_lane = None
        self.highlighted_tracked_lane_at = 0
        self.current_label_key = "curr"
        self.curr_token = None

        os.makedirs(args.nuscenes_save_dir, exist_ok=True)
        os.makedirs(args.nuscenes_preview_dir, exist_ok=True)
    
    def setup_ui(self):
        self.app = QApplication(sys.argv)
        self.window = QWidget()
        self.window.setWindowTitle("NuScenes Annotator & Visualizer")
        self.window.setWindowIcon(QIcon('nusc_icon.png'))  # Replace with the actual path to your PNG file
        self.window.setGeometry(0, 0, 960, 800)  # setting  the geometry of window
        self.setup_widgets()
        self.setup_layouts()
        self.setup_bindings()

    def setup_widgets(self):
        # checkboxs
        self.width0 = width0 = 120
        self.width1 = width1 = 220
        self.checkbox_use_mini = QCheckBox('Is Mini')
        self.checkbox_use_mini.setChecked(True)
        self.checkbox_use_mini.setFixedWidth(width0)

        self.label_viz = QLabel("Viz for...")
        self.checkbox_viz_curr = QCheckBox('Curr')
        self.checkbox_viz_curr.setChecked(True)
        self.checkbox_viz_left = QCheckBox('Left')
        self.checkbox_viz_left.setChecked(True)
        self.checkbox_viz_right = QCheckBox('Right')
        self.checkbox_viz_right.setChecked(True)
    
        self.button_group_checkbox_viz = QButtonGroup()
        self.button_group_checkbox_viz.setExclusive(False)
        self.button_group_checkbox_viz.addButton(self.checkbox_viz_curr)
        self.button_group_checkbox_viz.addButton(self.checkbox_viz_left)
        self.button_group_checkbox_viz.addButton(self.checkbox_viz_right)

        # create radio buttons
        self.label_label = QLabel("Label for...")
        self.radio_button1 = QRadioButton("Curr")
        self.radio_button2 = QRadioButton("Left")
        self.radio_button3 = QRadioButton("Right")

        self.radio_button1.setChecked(True)
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radio_button1)
        self.button_group.addButton(self.radio_button2)
        self.button_group.addButton(self.radio_button3)
        
        # buttons
        self.combobox_highlevel = QComboBox()
        self.options = ["Lane-keeping", "Left-lane-change", "Right-lane-change", "Stop sign", "Traffic light"]
        self.reverse_high_level_d = {k:i for i,k in enumerate(self.options)}
        self.reverse_high_level_d[None]=0
        self.combobox_highlevel.addItems(self.options)
        self.button_load_data = QPushButton("Load Nuscenes")
        self.button_load_annotation = QPushButton("Load Anno.")
        self.button_save_data = QPushButton("Save Anno.")
        self.button_load_data.setFixedWidth(width0)
        self.button_save_data.setFixedWidth(width0)
        
        self.button_clear = QPushButton("Clear")
        self.button_clear_all = QPushButton("Clear all")
        self.button_delete = QPushButton("Del")
        self.button_move_up = QPushButton("Up")
        self.button_move_down = QPushButton("Down")
        self.button_keyframe_add = QPushButton("Add frame")
        self.button_keyframe_del = QPushButton("Del frame")
        self.button_group_clear = QButtonGroup()
        self.button_group_clear.addButton(self.button_clear)
        self.button_group_clear.addButton(self.button_clear_all)
        self.button_group_move = QButtonGroup()
        self.button_group_move.addButton(self.button_delete)
        self.button_group_move.addButton(self.button_move_up)
        self.button_group_move.addButton(self.button_move_down)
        self.button_group_keyframe = QButtonGroup()
        self.button_group_keyframe.addButton(self.button_keyframe_add)
        self.button_group_keyframe.addButton(self.button_keyframe_del)

        self.button_clear.setFixedWidth(width1 / 2.5)
        self.button_clear_all.setFixedWidth(width1 / 2.5)
        self.button_delete.setFixedWidth(width1 / 3.2)
        self.button_move_up.setFixedWidth(width1 / 3.2)
        self.button_move_down.setFixedWidth(width1 / 3.2)
        self.button_keyframe_add.setFixedWidth(width1 / 2.5)
        self.button_keyframe_del.setFixedWidth(width1 / 2.5)

        self.canvas_widget = CanvasWidget()
        self.canvas_widget.setFixedSize(800, 800)

        self.label_legend = QLabel('')
        self.textedit_stats = QTextEdit()
        self.tableview_records = QTableView()
        self.tableview_records.setSizePolicy(width1, QSizePolicy.Expanding)
        self.tableview_records.setFixedWidth(width1)
        self.model_records = QStandardItemModel()

        self.tableview_lane_tokens = QTableView()
        self.model_lane_tokens = QStandardItemModel()

        self.tableview_tracked = QTableWidget()
        self.tableview_tracked.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableview_tracked.setFixedWidth(width1)
        self.tableview_tracked.setColumnCount(3)
        self.tableview_tracked.setHorizontalHeaderLabels(["Curr", "Left", "Right"])
        widget_width = self.tableview_tracked.width()
        column_width = widget_width // 3.5
        for column in range(3):
            self.tableview_tracked.setColumnWidth(column, column_width)
        
        self.tableview_tracked.setRowCount(100)
        for row in range(100):
            self.tableview_tracked.setRowHeight(row, 10)
        
        self.slider_ego_state = QSlider(orientation=1) # 1 corresponds to horizontal orientation
        self.canvas_label = QLabel('Scene Canvas')
        self.panel_label = QLabel('Control Panel')
        self.label_data_option = QLabel('Dataset option')
        self.record_label = QLabel("Record Tokens")
        self.lane_tokens_label = QLabel("Lane Tokens")
        self.highlevel_label = QLabel("Highlevel behavior")
        self.tracked_label = QLabel("Tracked Lanes")
        self.slider_label = QLabel('Timestep:')
    
    def setup_layouts(self):
        self.layout = QVBoxLayout()
        self.main_layout = QHBoxLayout()
        self.stats_record_layout = QVBoxLayout()
        self.info_layout = QVBoxLayout()
        self.slider_layout = QHBoxLayout()
        self.canvas_layout = QVBoxLayout()
        self.panel_layout = QVBoxLayout()
        self.stats_layout = QVBoxLayout()
        self.radio_group_layout = QVBoxLayout()       
        self.button_group_keyframe_layout = QHBoxLayout()
        self.button_group_clear_layout = QHBoxLayout()
        self.button_group_move_layout = QHBoxLayout()

        self.window.setLayout(self.layout)
        self.layout.addLayout(self.main_layout)
        self.layout.addLayout(self.info_layout)

        self.main_layout.addLayout(self.stats_record_layout)
        self.main_layout.addLayout(self.canvas_layout)
        self.main_layout.addLayout(self.panel_layout)    

        self.info_layout.addLayout(self.slider_layout)
        self.info_layout.addLayout(self.stats_layout)    
        
        self.stats_record_layout.addWidget(self.record_label)
        self.stats_record_layout.addWidget(self.tableview_records)
        self.stats_record_layout.addWidget(self.lane_tokens_label)
        self.stats_record_layout.addWidget(self.tableview_lane_tokens)        
        self.stats_record_layout.addWidget(self.highlevel_label)
        self.stats_record_layout.addWidget(self.combobox_highlevel)
        self.stats_record_layout.addWidget(self.tracked_label)

        self.button_group_keyframe_layout.addWidget(self.button_keyframe_add)
        self.button_group_keyframe_layout.addWidget(self.button_keyframe_del)
        self.stats_record_layout.addLayout(self.button_group_keyframe_layout)

        self.button_group_clear_layout.addWidget(self.button_clear)
        self.button_group_clear_layout.addWidget(self.button_clear_all)
        self.stats_record_layout.addLayout(self.button_group_clear_layout)
        
        self.button_group_move_layout.addWidget(self.button_delete)
        self.button_group_move_layout.addWidget(self.button_move_up)
        self.button_group_move_layout.addWidget(self.button_move_down)
        self.stats_record_layout.addLayout(self.button_group_move_layout)
        self.stats_record_layout.addWidget(self.tableview_tracked)

        self.canvas_layout.addWidget(self.canvas_label, alignment=Qt.AlignTop)
        self.canvas_layout.addWidget(self.canvas_widget)

        self.panel_layout.addWidget(self.panel_label, alignment=Qt.AlignTop)
        self.panel_layout.addWidget(self.label_data_option)
        self.panel_layout.addWidget(self.checkbox_use_mini)
        self.panel_layout.addWidget(self.button_load_data)
        self.panel_layout.addWidget(self.label_viz)
        self.panel_layout.addWidget(self.checkbox_viz_curr)
        self.panel_layout.addWidget(self.checkbox_viz_left)
        self.panel_layout.addWidget(self.checkbox_viz_right)
        self.panel_layout.addWidget(self.label_label)
        self.panel_layout.addLayout(self.radio_group_layout)
        self.radio_group_layout.addWidget(self.radio_button1)
        self.radio_group_layout.addWidget(self.radio_button2)
        self.radio_group_layout.addWidget(self.radio_button3)
        self.panel_layout.addStretch()
        self.panel_layout.addWidget(self.button_load_annotation)
        self.panel_layout.addStretch()
        self.panel_layout.addWidget(self.button_save_data)
        self.panel_layout.addStretch()
        self.panel_layout.addWidget(self.label_legend)

        self.slider_label.setFixedWidth(self.width1/2)
        self.slider_layout.addWidget(self.slider_label, alignment=Qt.AlignTop)
        self.slider_layout.addWidget(self.slider_ego_state)
        self.stats_layout.addWidget(self.textedit_stats)
    
    def setup_bindings(self):
        self.window.keyPressEvent = self.keyPressEvent
        self.window.keyReleaseEvent = self.keyReleaseEvent
        self.canvas_widget.doubleClicked.connect(self.on_canvas_double_clicked)
        self.canvas_widget.photoClicked.connect(self.on_canvas_clicked)
        self.button_load_data.clicked.connect(self.on_button_load_data_clicked)
        self.button_load_annotation.clicked.connect(self.on_button_load_annotation_clicked)
        self.button_save_data.clicked.connect(self.on_button_save_data_clicked)
        self.button_group.buttonClicked.connect(self.handleRadioButtonChange)
        self.button_group_clear.buttonClicked.connect(self.on_button_group_clear_clicked)
        self.button_group_move.buttonClicked.connect(self.on_button_group_move_clicked)
        self.button_group_keyframe.buttonClicked.connect(self.on_button_group_keyframe_clicked)
        self.tableview_records.clicked.connect(self.on_tableview_record_clicked)
        self.tableview_lane_tokens.clicked.connect(self.on_tableview_lane_tokens_clicked)
        self.tableview_tracked.clicked.connect(self.on_tableview_tracked_clicked)
        self.button_group_checkbox_viz.buttonClicked.connect(self.update_scene_func)
        self.slider_ego_state.valueChanged.connect(self.slider_ego_state_value_changed)
        self.combobox_highlevel.currentIndexChanged.connect(self.update_highlevel_label)


    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Control, 16777299]:
            self.ctrl_pressed = True

    def keyReleaseEvent(self, event):
        if event.key() in [Qt.Key_Control, 16777299]:
            self.ctrl_pressed = False

    def handleRadioButtonChange(self, button):
        if button.text() in ["Curr", "Left", "Right"]:
            self.current_label_key = button.text().lower()
        else:
            raise NotImplementedError

    def update_highlevel_label(self, index):
        if self.is_loaded:
            self.get_proper_frame()["high_level"] = self.options[index]
            self.textedit_stats.setText("Label highlevel as:%s"%(self.options[index]))

    def update_scene_func(self):
        self.update_scene()

    def update_scene(self, data=None):
        if self.is_loaded:
            self.my_painter = MyPainter(self.qimage_cache)
            
            # plot ego vehicle
            if self.ego_x_pixel != None:
                self.my_painter.plot_a_point(self.ego_x_pixel, self.ego_y_pixel, Qt.blue, stroke=20)
            
            # plot hover point
            if self.hover_x != None:
                self.my_painter.plot_a_point(self.hover_x, self.hover_y, QColor(0, 255, 0, 128), stroke=40, )
            
            # plot the centerlines nearby
            if self.plot_lanes is not None:
                pen = QPen(QColor(153, 0, 53, 183), 5, Qt.SolidLine, cap=Qt.RoundCap)
                for lane in self.plot_lanes:
                    xs_pixel, ys_pixel = self.my_tf.world_to_pixel(lane[-1][:,0], lane[-1][:,1])
                    self.my_painter.plot_line(xs_pixel, ys_pixel, pen=pen, arrow=True)
        
            # plot the highlighted selected lines
            if self.highlighted_lane is not None:
                pen = QPen(QColor(23, 0, 153, 183), 7, Qt.SolidLine, cap=Qt.RoundCap)
                xs_pixel, ys_pixel = self.my_tf.world_to_pixel(self.highlighted_lane[-1][:,0], self.highlighted_lane[-1][:,1])
                self.my_painter.plot_line(xs_pixel, ys_pixel, pen=pen, arrow=True)
            
            # plot the tracked/annotated lines
            lane_color = {
                "curr":  QColor(0, 255, 255, 180), 
                "left":  QColor(255, 0, 255, 180), 
                "right": QColor(255, 255, 0, 180)}
            viz_checks = {
                "curr": self.checkbox_viz_curr.isChecked(),
                "left": self.checkbox_viz_left.isChecked(),
                "right": self.checkbox_viz_right.isChecked(),
            }
            for key in viz_checks:
                if viz_checks[key]:
                    pen = QPen(lane_color[key], 12, Qt.SolidLine, cap=Qt.RoundCap)
                    tracked_lanes = self.get_proper_frame(data)["lanes"]
                    for lane_i, lane in enumerate(tracked_lanes[key]):
                        xs_pixel, ys_pixel = self.my_tf.world_to_pixel(lane[-1][:,0], lane[-1][:,1])
                        self.my_painter.plot_line(xs_pixel, ys_pixel, pen=pen, arrow=(lane_i==len(tracked_lanes[key])-1))

            # plot the highlighted annotated lines from the table
            lane_color_heavy = [
                QColor(0, 255, 255, 230), 
                QColor(255, 0, 255, 230), 
                QColor(255, 255, 0, 230)]
            if self.highlighted_tracked_lane:
                pen = QPen(lane_color_heavy[self.highlighted_tracked_lane_at], 16, Qt.SolidLine, cap=Qt.RoundCap)
                xs_pixel, ys_pixel = self.my_tf.world_to_pixel(self.highlighted_tracked_lane[-1][:,0], self.highlighted_tracked_lane[-1][:,1])
                self.my_painter.plot_line(xs_pixel, ys_pixel, pen=pen, arrow=True)

            # plot the scribbled lines
            self.canvas_widget.setPhoto(self.my_painter.pixmap, dont_fit_view=True)

            highlevel = self.get_proper_frame(data)["high_level"]
            self.combobox_highlevel.setCurrentIndex(self.reverse_high_level_d[highlevel])

    def update_table(self, data=None):       
        self.tableview_tracked.clearContents()
        tracked_lanes = self.get_proper_frame(data)["lanes"]
        for key_i, key in enumerate(["curr", "left", "right"]):
            for lane_i, lane in enumerate(tracked_lanes[key]):
                self.tableview_tracked.setItem(lane_i, key_i, QTableWidgetItem(lane[1]))
        self.tableview_tracked.update()

    def slider_ego_state_value_changed(self):
        if self.is_loaded:
            self.cur_ti = self.slider_ego_state.value()
            
            if self.cur_ti==0:
                self.button_keyframe_add.setEnabled(False)
                self.button_keyframe_del.setEnabled(False)
            else:
                if self.cur_ti in sorted(self.annotated_data.keys()):
                    self.button_keyframe_add.setEnabled(False)
                    self.button_keyframe_del.setEnabled(True)
                else:
                    self.button_keyframe_add.setEnabled(True)
                    self.button_keyframe_del.setEnabled(False)
            self.slider_label.setText(f'Timestep: {self.cur_ti}')
            state = self.ego_traj[self.cur_ti]
            self.ego_x_pixel, self.ego_y_pixel = self.my_tf.world_to_pixel(state[0], state[1])
            self.update_scene()
            self.update_table()

    def on_canvas_clicked(self, point):
        if self.is_loaded:
            x, y = self.my_tf.pixel_to_world(point.x(), point.y())
            pt = np.array([[x, y]])
            if self.plot_lanes is not None and len(self.plot_lanes)>0:
                d_min = 100000
                lane_min = None
                id_min = None
                self.textedit_stats.setText("current cursor:%.3f %.3f"%(x, y))
                for lane_i, lane in enumerate(self.plot_lanes):
                    d = np.min(np.linalg.norm(lane[-1][:,:2]-pt, axis=1))
                    if d < 4:
                        if d < d_min:
                            d_min = d
                            lane_min = lane
                            id_min = lane_i
                col_idx_dict={"curr":0, "left":1, "right":2}
                if lane_min is not None:
                    self.highlighted_lane = lane_min
                    self.tableview_lane_tokens.selectRow(id_min)
                    if self.ctrl_pressed:
                        key = self.current_label_key
                        tracked_lanes = self.get_proper_frame()["lanes"]
                        tracked_token_names = [xx[1] for xx in tracked_lanes[key]]
                        if lane_min[1] not in tracked_token_names:
                            tracked_lanes[key].append(lane_min)
                        else:
                            for ii in range(len(tracked_lanes[key])):
                                if tracked_lanes[key][ii][1] == lane_min[1]:
                                    break
                            del tracked_lanes[key][ii]
                    self.update_scene()
                    self.update_table()

    def on_canvas_double_clicked(self, point):
        if self.is_loaded:
            if self.ctrl_pressed:
                return
            self.hover_x = point.x()
            self.hover_y = point.y()

            # check lane records
            x, y = self.my_tf.pixel_to_world(self.hover_x, self.hover_y)
            tt1=time.time()
            self.plot_lanes = get_lanes_nearby(self.nusc_map, x, y, radius=6)
            print("Query took %.3f seconds"%(time.time()-tt1))
            
            # listview records
            self.model_lane_tokens.clear()
            self.model_lane_tokens.setHorizontalHeaderLabels(['LaneToken', 'Dist'])
            for item_tuple in self.plot_lanes:
                item = [QStandardItem("%s"%(item_tuple[1])),
                        QStandardItem("%.3f"%(item_tuple[0]))]
                self.model_lane_tokens.appendRow(item)
            self.tableview_lane_tokens.setModel(self.model_lane_tokens)
            self.tableview_lane_tokens.update()
            self.highlighted_lane = None
            self.update_scene()

    def get_proper_frame(self, data=None):
        for ti in range(self.cur_ti,-1,-1):
            if data is not None:
                if ti in data:
                    return data[ti]
            else:
                if ti in self.annotated_data:
                    return self.annotated_data[ti]    

    def on_button_group_clear_clicked(self, button):
        if self.is_loaded:
            keyframe = self.get_proper_frame()
            if button.text() == "Clear":
                keyframe["lanes"][self.current_label_key] = []
            elif button.text() == "Clear all":
                keyframe["lanes"] = {"curr":[], "left":[], "right":[]}
            self.highlighted_tracked_lane = None
            self.highlighted_tracked_lane_at = 0
            self.update_scene()
            self.update_table()
        
    def on_button_group_move_clicked(self, button):
        if self.is_loaded:
            selected_items = self.tableview_tracked.selectedItems()
            mode = button.text()
            if selected_items:
                assert len(selected_items)==1
                key_list=["curr", "left", "right"]
                for item in selected_items:
                    lane_token = self.tableview_tracked.item(item.row(), item.column()).text()
                    if lane_token is not None and len(lane_token)>0:
                        key = key_list[item.column()]
                        tracked_lanes_key = self.get_proper_frame()["lanes"][key]
                        for i,lane in enumerate(tracked_lanes_key):
                            if lane[1] == lane_token:
                                if mode=="Del":
                                    del tracked_lanes_key[i]
                                elif mode=="Up":
                                    if i!=0:
                                        tracked_lanes_key[i], tracked_lanes_key[i-1] = tracked_lanes_key[i-1], tracked_lanes_key[i]
                                elif mode=="Down":
                                    if i!=len(tracked_lanes_key)-1:
                                        tracked_lanes_key[i], tracked_lanes_key[i+1] = tracked_lanes_key[i+1], tracked_lanes_key[i]
                                else:
                                    raise NotImplementedError
                                break
            self.update_scene()
            self.update_table()

    def on_button_group_keyframe_clicked(self, button):
        if self.is_loaded:
            if button.text()=="Add frame":
                self.annotated_data[self.cur_ti]={"high_level": None, "lanes":{"curr":[], "left":[], "right":[]}}
                print("Now annotated frames are", self.annotated_data.keys())
                self.button_keyframe_add.setEnabled(False)
                self.button_keyframe_del.setEnabled(True)
            elif button.text()=="Del frame":
                del self.annotated_data[self.cur_ti]
                for ti in range(self.cur_ti-1,-1,-1):
                    if ti in self.annotated_data:
                        break
                self.button_keyframe_add.setEnabled(True)
                self.button_keyframe_del.setEnabled(False)
                
                print("Now annotated frames are", self.annotated_data.keys())
            else:
                raise NotImplementedError
            self.update_scene()
            self.update_table()
    
    # load the nuscenes data
    def on_button_load_data_clicked(self):
        if self.checkbox_use_mini.isChecked():
            version='v1.0-mini'
            dataroot=os.path.join(args.nuscenes_data_dir, 'nuscenes_mini')
        else:
            version='v1.0-trainval'
            dataroot=os.path.join(args.nuscenes_data_dir, 'nuscenes')
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        nusc_map_d={}
        location_list=["boston-seaport", "singapore-hollandvillage", "singapore-queenstown", "singapore-onenorth"]
        for map_name in location_list:
            nusc_map_d[map_name] = NuScenesMap(os.path.join(args.nuscenes_data_dir, 'nuscenes'), map_name=map_name)
        self.nusc = nusc
        self.nusc_map_d = nusc_map_d
        self.location_list = location_list

        # update the record tokens
        scene_list = []
        for scene_id,scene in enumerate(self.nusc.scene):
            log = self.nusc.get("log", scene["log_token"])
            the_token = scene["first_sample_token"]
            scene_list.append([str(the_token), str(scene["log_token"]), str(log["location"])])

        print("Scene_list length:", len(scene_list))
        self.model_records.setHorizontalHeaderLabels(['Scene token0', 'Log token', 'Location'])
        for item_i, item_tuple in enumerate(scene_list):
            item = [QStandardItem(item_text) for item_text in item_tuple]
            self.model_records.appendRow(item)

        self.tableview_records.setModel(self.model_records)
        self.tableview_records.update()

        for item_i, item_tuple in enumerate(scene_list):
            if os.path.exists("%s/%s.pickle"%(args.nuscenes_save_dir, item_tuple[0])):
                self.bold_row(item_i)

        self.button_load_data.setEnabled(False)
        self.is_loaded = True
        self.curr_token = None
        self.scene_id = None
        self.hover_x = None
        self.hover_y = None
        self.ego_x_pixel = None
        self.ego_y_pixel = None
        self.plot_lanes = None
        self.highlighted_lane = None
        self.highlighted_tracked_lane = None
        self.highlighted_tracked_lane_at = 0

        self.viz_scene(scene_id=0, ti=0)
        self.update_scene()
        self.update_table()

    def viz_scene(self, scene_id=0, ti=0):
        # render the first record
        if scene_id==self.scene_id and ti==self.cur_ti:
            return
        self.scene_id = scene_id
        self.cur_ti = ti
        
        my_scene = self.nusc.scene[self.scene_id]
        location = self.nusc.get("log", my_scene["log_token"])["location"]
        self.nusc_map = self.nusc_map_d[location]
        self.curr_token = my_scene["first_sample_token"]
        the_token = self.curr_token
        ego_traj = []
        while the_token != "":
            the_sample = self.nusc.get("sample", the_token)
            the_lidar_data = self.nusc.get("sample_data", the_sample['data']["LIDAR_TOP"])
            the_xy = self.nusc.get("ego_pose", the_lidar_data["ego_pose_token"])["translation"]
            ego_traj.append(the_xy)
            the_token = the_sample["next"]
        self.ego_traj=np.array(ego_traj)
        
        # responding to the variables
        self.slider_ego_state.setValue(0)
        self.slider_ego_state.setRange(0, self.ego_traj.shape[0]-1)
        self.slider_ego_state.setTickPosition(QSlider.TicksBelow)
        self.slider_ego_state.setTickInterval(1)

        if self.curr_token in self.cache:
            self.qimage_cache = self.cache[self.curr_token]["qimage_cache"]
            self.qimage_legend_cache = self.cache[self.curr_token]["qimage_legend_cache"]
            self.my_tf = self.cache[self.curr_token]["my_tf"]
            self.canvas_widget.setPhoto(QPixmap(self.qimage_cache)) 
            self.label_legend.setPixmap(QPixmap(self.qimage_legend_cache).scaledToWidth(self.width0))
        else:                        
            # plot the bird-eye-view scenes
            # get img outer coordinates in ego/world-frame
            fig, handles, labels, xmin, xmax, ymin, ymax = visualize_nuscenes_scene(self.nusc_map, self.ego_traj)
            canvas = FigureCanvas(fig)
            canvas.draw()
            self.qimage_cache = QImage(canvas.buffer_rgba(), canvas.size().width(), canvas.size().height(), QImage.Format_ARGB32)
            pixmap = QPixmap(self.qimage_cache)
            self.canvas_widget.setPhoto(pixmap)
            self.my_tf = MyTF(xmin, xmax, ymin, ymax, pixmap.width(), pixmap.height())
            
            # plot the labels on the right
            fig = visualize_nuscenes_legends(handles, labels)
            canvas = FigureCanvas(fig)
            canvas.draw()
            im_legend = QImage(canvas.buffer_rgba(), canvas.size().width(), canvas.size().height(), QImage.Format_ARGB32)
            self.qimage_legend_cache = remove_qimage_margin(im_legend)
            pixmap = QPixmap(self.qimage_legend_cache).scaledToWidth(self.width0)
            self.label_legend.setPixmap(pixmap)
            self.cache[self.curr_token] = {
                "qimage_cache": self.qimage_cache, 
                "my_tf": self.my_tf,
                "qimage_legend_cache": self.qimage_legend_cache
            }   
        self.slider_ego_state_value_changed()
        self.reset_data()

    def reset_data(self):
        assert self.cur_ti==0 and self.curr_token in self.cache
        if "annotated_data" in self.cache[self.curr_token]:
            print("load from cache")
            self.annotated_data = self.cache[self.curr_token]["annotated_data"]
        else: # new data
            annot_data_path = "%s/%s.pickle"%(args.nuscenes_save_dir, self.curr_token)
            if os.path.exists(annot_data_path):
                with open(annot_data_path, "rb") as f:
                    self.annotated_data = pickle.load(f)
            else:
                print("create new")
                self.annotated_data = {0:{"high_level": None, "lanes":{"curr":[], "left":[], "right":[]}}}
            self.cache[self.curr_token]["annotated_data"] = self.annotated_data

    def bold_row(self, row_idx):
        model = self.tableview_records.model()
        item = model.item(row_idx, 0)
        font = item.font()
        font.setBold(True)
        item.setFont(font)

    def on_tableview_record_clicked(self):
        self.hover_x = None
        self.hover_y = None
        self.plot_lanes = None
        self.highlighted_lane = None
        self.model_lane_tokens.clear()
        self.highlighted_tracked_lane = None
        self.highlighted_tracked_lane_at = 0
        selection_model = self.tableview_records.selectionModel()
        if selection_model.hasSelection():
            # Get the current index (selected cell) in the table view
            current_index = selection_model.currentIndex()
            # Extract the row ID from the model data
            scene_id = current_index.row()
            token = self.model_records.item(current_index.row(), 0).text()
            self.textedit_stats.setText("Selected scene_id:%s token:%s"%(scene_id, token))
            self.radio_button1.setChecked(True)
            self.current_label_key = "curr"
            self.viz_scene(scene_id=scene_id, ti=0)
            self.update_scene()
            self.update_table()
            
    def on_tableview_lane_tokens_clicked(self):
        selection_model = self.tableview_lane_tokens.selectionModel()
        if selection_model.hasSelection():
            current_index = selection_model.currentIndex()
            if self.plot_lanes is not None:
                self.highlighted_lane = self.plot_lanes[current_index.row()]
            self.update_scene()
            self.update_table()

    def on_tableview_tracked_clicked(self):
        self.highlighted_tracked_lane = None
        self.highlighted_tracked_lane_at = 0
        selected_items = self.tableview_tracked.selectedItems()
        if selected_items:
            assert len(selected_items)==1
            for item in selected_items:
                key_list = ["curr", "left", "right"]
                tracked_lanes_key = self.get_proper_frame()["lanes"][key_list[item.column()]]
                self.highlighted_tracked_lane = tracked_lanes_key[item.row()]
                lane_token = self.tableview_tracked.item(item.row(), item.column()).text()
                assert self.highlighted_tracked_lane[1]==lane_token
                self.highlighted_tracked_lane_at = item.column()
        self.update_scene()

    def on_button_load_annotation_clicked(self):
        if self.is_loaded and self.curr_token is not None:
            data_path="%s/%s.pickle"%(args.nuscenes_preview_dir, self.curr_token)
            if os.path.exists(data_path):
                with open(data_path, "rb") as f:
                    annotated_data = pickle.load(f)
                    self.update_scene(data=annotated_data)
                    self.update_table(data=annotated_data)
            else:
                message_box = QMessageBox()
                message_box.setIcon(QMessageBox.Warning)
                message_box.setWindowTitle("Warning")
                message_box.setText("Not found:%s!"%(data_path))
                message_box.setStandardButtons(QMessageBox.Ok)
                message_box.exec_()

    def on_button_save_data_clicked(self):
        if self.is_loaded:
            with open("%s/%s.pickle"%(args.nuscenes_save_dir, self.curr_token), "wb") as f:
                pickle.dump(self.annotated_data, f)
            self.bold_row(self.scene_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NuScenes Visualizer and Annotation Tool v1.0")
    parser.add_argument("--nuscenes_data_dir", type=str, default="../../dataset")
    parser.add_argument("--nuscenes_preview_dir", type=str, default="./preview_data")
    parser.add_argument("--nuscenes_save_dir", type=str, default="./saved_data")
    args = parser.parse_args()
    my_gui_app = MyGUIApp()
    my_gui_app.window.show()
    sys.exit(my_gui_app.app.exec_())
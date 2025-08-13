import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QTextEdit, QGroupBox, QGridLayout, QRadioButton,
                               QButtonGroup, QScrollArea, QFrame, QSplitter,
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from controller import Controller
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class VisualizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        controls_layout = QHBoxLayout()
        self.reset_view_btn = QPushButton("Reset View")
        self.toggle_grid_btn = QPushButton("Toggle Grid")
        self.clear_plot_btn = QPushButton("Clear Plot")
        
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.toggle_grid_btn.clicked.connect(self.toggle_grid)
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        
        controls_layout.addWidget(self.reset_view_btn)
        controls_layout.addWidget(self.toggle_grid_btn)
        controls_layout.addWidget(self.clear_plot_btn)
        controls_layout.addStretch()
        
        self.layout.addLayout(controls_layout)
        
        self.is_3d = False
        self.ax = None
        self.grid_visible = True
        
    def setup_plot(self, is_3d):
        self.is_3d = is_3d
        self.figure.clear()
        
        if is_3d:
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X', fontsize=12, color='white')
            self.ax.set_ylabel('Y', fontsize=12, color='white')
            self.ax.set_zlabel('Z', fontsize=12, color='white')
            self.ax.set_title('3D Point Transformation', fontsize=14, color='white', pad=20)
        else:
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel('X', fontsize=12, color='white')
            self.ax.set_ylabel('Y', fontsize=12, color='white')
            self.ax.set_title('2D Point Transformation', fontsize=14, color='white', pad=20)
            self.ax.set_aspect('equal', adjustable='box')
        
        self.figure.patch.set_facecolor('#2c3e50')
        self.ax.set_facecolor('#34495e')
        self.ax.tick_params(colors='white')
        self.ax.grid(self.grid_visible, alpha=0.3, color='white')
        
        if is_3d:
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('white')
            self.ax.yaxis.pane.set_edgecolor('white')
            self.ax.zaxis.pane.set_edgecolor('white')
            self.ax.xaxis.pane.set_alpha(0.1)
            self.ax.yaxis.pane.set_alpha(0.1)
            self.ax.zaxis.pane.set_alpha(0.1)
        
        self.canvas.draw()
    
    def plot_points(self, original_points, transformed_points=None):
        if self.ax is None:
            return
            
        self.ax.clear()
        
        self.ax.set_facecolor('#34495e')
        self.ax.tick_params(colors='white')
        self.ax.grid(self.grid_visible, alpha=0.3, color='white')
        
        if self.is_3d:
            self.ax.set_xlabel('X', fontsize=12, color='white')
            self.ax.set_ylabel('Y', fontsize=12, color='white')
            self.ax.set_zlabel('Z', fontsize=12, color='white')
            self.ax.set_title('3D Point Transformation', fontsize=14, color='white', pad=20)
            
            if len(original_points) > 0:
                orig_x = [p.x for p in original_points if hasattr(p, 'x')]
                orig_y = [p.y for p in original_points if hasattr(p, 'y')]
                orig_z = [p.z for p in original_points if hasattr(p, 'z')]
                
                if orig_x and orig_y and orig_z:
                    self.ax.scatter(orig_x, orig_y, orig_z, c='#3498db', s=100, 
                                  alpha=0.8, label='Original Points', marker='o')
                    
                    if len(orig_x) > 1:
                        self.ax.plot(orig_x, orig_y, orig_z, c='#3498db', alpha=0.5, linewidth=2)
            
            if transformed_points is not None and transformed_points.shape[0] >= 3:
                trans_x = transformed_points[0, :]
                trans_y = transformed_points[1, :]
                trans_z = transformed_points[2, :]
                
                self.ax.scatter(trans_x, trans_y, trans_z, c='#e74c3c', s=100, 
                              alpha=0.8, label='Transformed Points', marker='^')
                
                if len(trans_x) > 1:
                    self.ax.plot(trans_x, trans_y, trans_z, c='#e74c3c', alpha=0.5, linewidth=2)
                
                if len(original_points) > 0 and len(orig_x) == len(trans_x):
                    for i in range(len(orig_x)):
                        self.ax.plot([orig_x[i], trans_x[i]], [orig_y[i], trans_y[i]], 
                                   [orig_z[i], trans_z[i]], c='#f39c12', alpha=0.6, linewidth=1.5)
            
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('white')
            self.ax.yaxis.pane.set_edgecolor('white')
            self.ax.zaxis.pane.set_edgecolor('white')
            self.ax.xaxis.pane.set_alpha(0.1)
            self.ax.yaxis.pane.set_alpha(0.1)
            self.ax.zaxis.pane.set_alpha(0.1)
            
        else:  
            self.ax.set_xlabel('X', fontsize=12, color='white')
            self.ax.set_ylabel('Y', fontsize=12, color='white')
            self.ax.set_title('2D Point Transformation', fontsize=14, color='white', pad=20)
            self.ax.set_aspect('equal', adjustable='box')
            
            if len(original_points) > 0:
                orig_x = [p.x for p in original_points if hasattr(p, 'x')]
                orig_y = [p.y for p in original_points if hasattr(p, 'y')]
                
                if orig_x and orig_y:
                    self.ax.scatter(orig_x, orig_y, c='#3498db', s=100, 
                                  alpha=0.8, label='Original Points', marker='o')
                    
                    if len(orig_x) > 1:
                        self.ax.plot(orig_x, orig_y, c='#3498db', alpha=0.5, linewidth=2)
            
            if transformed_points is not None and transformed_points.shape[0] >= 2:
                trans_x = transformed_points[0, :]
                trans_y = transformed_points[1, :]
                
                self.ax.scatter(trans_x, trans_y, c='#e74c3c', s=100, 
                              alpha=0.8, label='Transformed Points', marker='^')
                
                if len(trans_x) > 1:
                    self.ax.plot(trans_x, trans_y, c='#e74c3c', alpha=0.5, linewidth=2)
                
                if len(original_points) > 0 and len(orig_x) == len(trans_x):
                    for i in range(len(orig_x)):
                        self.ax.annotate('', xy=(trans_x[i], trans_y[i]), xytext=(orig_x[i], orig_y[i]),
                                       arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2, alpha=0.7))
        
        if len(original_points) > 0 and transformed_points is not None:
            legend = self.ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('#34495e')
            legend.get_frame().set_alpha(0.8)
            for text in legend.get_texts():
                text.set_color('white')
        
        if len(original_points) > 0 or (transformed_points is not None and transformed_points.size > 0):
            margins = 0.1
            if self.is_3d:
                self.ax.margins(margins)
            else:
                self.ax.margins(margins)
        
        self.canvas.draw()
    
    def reset_view(self):
        if self.ax and self.is_3d:
            self.ax.view_init(elev=20, azim=45)
        self.canvas.draw()
    
    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        if self.ax:
            self.ax.grid(self.grid_visible, alpha=0.3, color='white')
            self.canvas.draw()
    
    def clear_plot(self):
        if self.ax:
            self.ax.clear()
            self.ax.set_facecolor('#34495e')
            self.ax.tick_params(colors='white')
            self.ax.grid(self.grid_visible, alpha=0.3, color='white')
            
            if self.is_3d:
                self.ax.set_xlabel('X', fontsize=12, color='white')
                self.ax.set_ylabel('Y', fontsize=12, color='white')
                self.ax.set_zlabel('Z', fontsize=12, color='white')
                self.ax.set_title('3D Point Transformation', fontsize=14, color='white', pad=20)
            else:
                self.ax.set_xlabel('X', fontsize=12, color='white')
                self.ax.set_ylabel('Y', fontsize=12, color='white')
                self.ax.set_title('2D Point Transformation', fontsize=14, color='white', pad=20)
                self.ax.set_aspect('equal', adjustable='box')
            
            self.canvas.draw()


class GeometricTransformGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geometric Transformer")
        self.setGeometry(100, 100, 900, 700)
        self.controller = Controller()
        self.is_3d_mode = False
        self.setupUI()
        self.setStyleSheet(self.get_stylesheet())
    
    def setupUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(splitter)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        title_label = QLabel("🔧 Geometric Transformer")
        title_label.setObjectName("title")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        
        mode_group = QGroupBox("Transformation Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup()
        self.mode_2d = QRadioButton("2D Mode")
        self.mode_3d = QRadioButton("3D Mode")
        self.mode_2d.setChecked(True)
        
        self.mode_button_group.addButton(self.mode_2d, 0)
        self.mode_button_group.addButton(self.mode_3d, 1)
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.mode_2d)
        mode_layout.addWidget(self.mode_3d)
        left_layout.addWidget(mode_group)
        
        points_group = QGroupBox("Point Management")
        points_layout = QVBoxLayout(points_group)
        
        add_point_layout = QGridLayout()
        
        self.x_input = QLineEdit("0.0")
        self.y_input = QLineEdit("0.0")
        self.z_input = QLineEdit("0.0")
        self.z_input.setEnabled(False)  # Disabled in 2D mode initially
        
        add_point_layout.addWidget(QLabel("X:"), 0, 0)
        add_point_layout.addWidget(self.x_input, 0, 1)
        add_point_layout.addWidget(QLabel("Y:"), 0, 2)
        add_point_layout.addWidget(self.y_input, 0, 3)
        add_point_layout.addWidget(QLabel("Z:"), 1, 0)
        add_point_layout.addWidget(self.z_input, 1, 1)
        
        self.add_point_btn = QPushButton("➕ Add Point")
        self.add_point_btn.clicked.connect(self.add_point)
        add_point_layout.addWidget(self.add_point_btn, 1, 2, 1, 2)
        
        points_layout.addLayout(add_point_layout)
        
        self.points_table = QTableWidget()
        self.points_table.setMaximumHeight(150)
        self.update_points_table()
        points_layout.addWidget(self.points_table)
        
        self.clear_points_btn = QPushButton("🗑️ Clear All Points")
        self.clear_points_btn.setObjectName("clear_button")
        self.clear_points_btn.clicked.connect(self.clear_points)
        points_layout.addWidget(self.clear_points_btn)
        
        left_layout.addWidget(points_group)
        
        transform_group = QGroupBox("Transformation Parameters")
        transform_layout = QGridLayout(transform_group)
        
        transform_layout.addWidget(QLabel("Scale:"), 0, 0)
        self.scale_input = QLineEdit("1.0")
        transform_layout.addWidget(self.scale_input, 0, 1)
        
        transform_layout.addWidget(QLabel("Angle (°):"), 1, 0)
        self.angle_input = QLineEdit("0.0")
        transform_layout.addWidget(self.angle_input, 1, 1)
        
        self.axis_label = QLabel("Rotation Axis:")
        self.axis_x_input = QLineEdit("0.0")
        self.axis_y_input = QLineEdit("0.0")
        self.axis_z_input = QLineEdit("1.0")
        
        transform_layout.addWidget(self.axis_label, 2, 0)
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("X:"))
        axis_layout.addWidget(self.axis_x_input)
        axis_layout.addWidget(QLabel("Y:"))
        axis_layout.addWidget(self.axis_y_input)
        axis_layout.addWidget(QLabel("Z:"))
        axis_layout.addWidget(self.axis_z_input)
        transform_layout.addLayout(axis_layout, 2, 1)
        
        transform_layout.addWidget(QLabel("Translation X:"), 3, 0)
        self.tx_input = QLineEdit("0.0")
        transform_layout.addWidget(self.tx_input, 3, 1)
        
        transform_layout.addWidget(QLabel("Translation Y:"), 4, 0)
        self.ty_input = QLineEdit("0.0")
        transform_layout.addWidget(self.ty_input, 4, 1)
        
        self.tz_label = QLabel("Translation Z:")
        self.tz_input = QLineEdit("0.0")
        transform_layout.addWidget(self.tz_label, 5, 0)
        transform_layout.addWidget(self.tz_input, 5, 1)
        
        self.set_3d_controls_visible(False)
        
        left_layout.addWidget(transform_group)
        
        self.perform_btn = QPushButton("🚀 Perform Transformation")
        self.perform_btn.setObjectName("calculate_button")
        self.perform_btn.clicked.connect(self.perform_transformation)
        left_layout.addWidget(self.perform_btn)
        
        preset_layout = QHBoxLayout()
        self.preset_rotate_btn = QPushButton("📐 Rotate 90°")
        self.preset_scale_btn = QPushButton("📏 Scale 2x")
        self.preset_translate_btn = QPushButton("📍 Translate (1,1)")
        
        self.preset_rotate_btn.clicked.connect(lambda: self.load_preset("rotate"))
        self.preset_scale_btn.clicked.connect(lambda: self.load_preset("scale"))
        self.preset_translate_btn.clicked.connect(lambda: self.load_preset("translate"))
        
        preset_layout.addWidget(self.preset_rotate_btn)
        preset_layout.addWidget(self.preset_scale_btn)
        preset_layout.addWidget(self.preset_translate_btn)
        left_layout.addLayout(preset_layout)
        
        left_layout.addStretch()
        
        results_title = QLabel("Results")
        results_title.setObjectName("title")
        results_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(results_title)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
    
    def get_stylesheet(self):
        return """
        QMainWindow {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #2c3e50, stop: 1 #34495e);
            color: #ecf0f1;
        }
        
        QLabel {
            color: #ecf0f1;
            font-size: 12px;
        }
        
        QGroupBox {
            font-size: 13px;
            font-weight: bold;
            color: #3498db;
            border: 2px solid #34495e;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 5px;
            background: rgba(52, 73, 94, 0.3);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 10px 0 10px;
            color: #3498db;
        }
        
        QLineEdit {
            background: rgba(44, 62, 80, 0.8);
            border: 2px solid #34495e;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 11px;
            color: #ecf0f1;
            min-height: 20px;
        }
        
        QLineEdit:focus {
            border: 2px solid #3498db;
            background: rgba(44, 62, 80, 1.0);
        }
        
        QLineEdit:hover {
            border: 2px solid #52c0f5;
        }
        
        QLineEdit:disabled {
            background: rgba(44, 62, 80, 0.4);
            color: #7f8c8d;
            border: 2px solid #2c3e50;
        }
        
        QPushButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #3498db, stop: 1 #2980b9);
            border: none;
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 11px;
            font-weight: bold;
            color: white;
            min-width: 100px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #52c0f5, stop: 1 #3498db);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #2980b9, stop: 1 #1f4e79);
        }
        
        QPushButton#calculate_button {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #27ae60, stop: 1 #229954);
            font-size: 13px;
            padding: 14px 25px;
        }
        
        QPushButton#calculate_button:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #2ecc71, stop: 1 #27ae60);
        }
        
        QPushButton#clear_button {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #e67e22, stop: 1 #d35400);
        }
        
        QPushButton#clear_button:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #f39c12, stop: 1 #e67e22);
        }
        
        QTextEdit {
            background: rgba(44, 62, 80, 0.9);
            border: 2px solid #34495e;
            border-radius: 8px;
            padding: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            color: #ecf0f1;
            line-height: 1.4;
        }
        
        QTextEdit:focus {
            border: 2px solid #3498db;
        }
        
        QTableWidget {
            background: rgba(44, 62, 80, 0.8);
            border: 2px solid #34495e;
            border-radius: 6px;
            font-size: 10px;
            color: #ecf0f1;
            gridline-color: #34495e;
        }
        
        QTableWidget::item {
            padding: 4px;
            border-bottom: 1px solid #34495e;
        }
        
        QTableWidget::item:selected {
            background: rgba(52, 152, 219, 0.3);
        }
        
        QHeaderView::section {
            background: rgba(52, 152, 219, 0.3);
            color: #ecf0f1;
            padding: 6px;
            border: none;
            font-weight: bold;
        }
        
        QRadioButton {
            color: #ecf0f1;
            font-size: 12px;
            spacing: 8px;
        }
        
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
        }
        
        QRadioButton::indicator:unchecked {
            border: 2px solid #34495e;
            border-radius: 8px;
            background: rgba(44, 62, 80, 0.8);
        }
        
        QRadioButton::indicator:checked {
            border: 2px solid #3498db;
            border-radius: 8px;
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #3498db, stop: 1 #2980b9);
        }
        
        QSplitter::handle {
            background: #34495e;
            width: 3px;
        }
        
        QSplitter::handle:hover {
            background: #3498db;
        }
        
        QWidget {
            background: transparent;
        }
        
        QLabel#title {
            color: #ecf0f1;
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 8px;
            margin: 5px;
        }
        """
    
    def on_mode_changed(self, button):
        self.is_3d_mode = (button == self.mode_3d)
        self.set_3d_controls_visible(self.is_3d_mode)
        self.z_input.setEnabled(self.is_3d_mode)
        self.update_points_table()
        self.clear_results()
    
    def set_3d_controls_visible(self, visible):
        self.axis_label.setVisible(visible)
        self.axis_x_input.setVisible(visible)
        self.axis_y_input.setVisible(visible)
        self.axis_z_input.setVisible(visible)
        self.tz_label.setVisible(visible)
        self.tz_input.setVisible(visible)
    
    def add_point(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            
            if self.is_3d_mode:
                self.controller.add3DPoint(x, y, z)
                self.results_text.append(f"Added 3D point: ({x:.2f}, {y:.2f}, {z:.2f})")
            else:
                self.controller.add2DPoint(x, y)
                self.results_text.append(f"Added 2D point: ({x:.2f}, {y:.2f})")
            
            self.update_points_table()
            
        except ValueError:
            self.results_text.append("Error: Please enter valid numbers for coordinates.")
        except Exception as e:
            self.results_text.append(f"Error adding point: {str(e)}")
    
    def clear_points(self):
        self.controller.current2DPoints.clear()
        self.controller.current3DPoints.clear()
        self.update_points_table()
        self.results_text.append("All points cleared.")
    
    def update_points_table(self):
        if self.is_3d_mode:
            points = self.controller.current3DPoints
            self.points_table.setColumnCount(3)
            self.points_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        else:
            points = self.controller.current2DPoints
            self.points_table.setColumnCount(2)
            self.points_table.setHorizontalHeaderLabels(["X", "Y"])
        
        self.points_table.setRowCount(len(points))
        
        for i, point in enumerate(points):
            if hasattr(point, 'x') and hasattr(point, 'y'):
                self.points_table.setItem(i, 0, QTableWidgetItem(f"{point.x:.2f}"))
                self.points_table.setItem(i, 1, QTableWidgetItem(f"{point.y:.2f}"))
                if self.is_3d_mode and hasattr(point, 'z'):
                    self.points_table.setItem(i, 2, QTableWidgetItem(f"{point.z:.2f}"))
        
        self.points_table.resizeColumnsToContents()
    
    def perform_transformation(self):
        try:
            scale = float(self.scale_input.text())
            angle = float(self.angle_input.text())
            tx = float(self.tx_input.text())
            ty = float(self.ty_input.text())
            
            points = self.controller.current3DPoints if self.is_3d_mode else self.controller.current2DPoints
            if not points:
                self.results_text.append("Error: No points to transform. Please add some points first.")
                return
            
            self.results_text.clear()
            self.results_text.append(f"=== {'3D' if self.is_3d_mode else '2D'} Transformation ===")
            self.results_text.append(f"Points to transform: {len(points)}")
            
            if self.is_3d_mode:
                tz = float(self.tz_input.text())
                axis_x = float(self.axis_x_input.text())
                axis_y = float(self.axis_y_input.text())
                axis_z = float(self.axis_z_input.text())
                
                self.results_text.append(f"Scale: {scale}")
                self.results_text.append(f"Rotation: {angle}° around axis ({axis_x:.2f}, {axis_y:.2f}, {axis_z:.2f})")
                self.results_text.append(f"Translation: ({tx:.2f}, {ty:.2f}, {tz:.2f})")
                
                result = self.controller.perform3D(axis_x, axis_y, axis_z, scale, angle, tx, ty, tz)
                
            else:
                self.results_text.append(f"Scale: {scale}")
                self.results_text.append(f"Rotation: {angle}°")
                self.results_text.append(f"Translation: ({tx:.2f}, {ty:.2f})")
                
                result = self.controller.perform2D(scale, angle, tx, ty)
            
            self.results_text.append("\n--- Transformed Points ---")
            if result is not None:
                for i in range(result.shape[1]):
                    if self.is_3d_mode:
                        x, y, z = result[0:3, i]
                        self.results_text.append(f"Point {i+1}: ({x:.4f}, {y:.4f}, {z:.4f})")
                    else:
                        x, y = result[0:2, i]
                        self.results_text.append(f"Point {i+1}: ({x:.4f}, {y:.4f})")
            else:
                self.results_text.append("No result returned from transformation.")
                
        except ValueError:
            self.results_text.append("Error: Please enter valid numbers for all parameters.")
        except Exception as e:
            self.results_text.append(f"Error during transformation: {str(e)}")
            self.results_text.append("Make sure you have imported P2, P3, and Quaternion classes.")
    
    def clear_results(self):
        self.results_text.clear()
    
    def load_preset(self, preset_type):
        if preset_type == "rotate":
            self.scale_input.setText("1.0")
            self.angle_input.setText("90.0")
            self.tx_input.setText("0.0")
            self.ty_input.setText("0.0")
            if self.is_3d_mode:
                self.tz_input.setText("0.0")
                self.axis_x_input.setText("0.0")
                self.axis_y_input.setText("0.0")
                self.axis_z_input.setText("1.0")
        elif preset_type == "scale":
            self.scale_input.setText("2.0")
            self.angle_input.setText("0.0")
            self.tx_input.setText("0.0")
            self.ty_input.setText("0.0")
            if self.is_3d_mode:
                self.tz_input.setText("0.0")
        elif preset_type == "translate":
            self.scale_input.setText("1.0")
            self.angle_input.setText("0.0")
            self.tx_input.setText("1.0")
            self.ty_input.setText("1.0")
            if self.is_3d_mode:
                self.tz_input.setText("1.0")

def main():
    app = QApplication(sys.argv)
    
    try:
        window = GeometricTransformGUI()
        window.show()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure to import your P2, P3, and Quaternion classes!")
        return
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

"""
GUI Components for Stars2Cells Viewer 
"""

import numpy as np
import time
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QFormLayout, 
                             QDialogButtonBox, QTextEdit, QMessageBox, QComboBox,
                             QPushButton, QProgressDialog, QLineEdit, QCheckBox,
                             QFileDialog, QApplication, QScrollArea, QWidget, QGroupBox) 
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import logging
import threading
import importlib

# Import centralized step info
from .step_info import *

from .config import PipelineConfig

class QTextEditLogger(logging.Handler):
    """Custom logging handler that emits log messages as Qt signals"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        
    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class InteractivePlot(pg.PlotWidget):
    """Custom plot widget with click selection support"""
    pointClicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter = None
        self.click_select_enabled = False
        self.point_positions = None
        self.deleted_indices = set()
        
    def mousePressEvent(self, event):
        """Handle mouse press for point selection"""
        if self.point_positions is not None and event.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(event.pos())
            x_click, y_click = pos.x(), pos.y()
            
            if len(self.point_positions) > 0:
                distances = np.sqrt((self.point_positions[:, 0] - x_click)**2 + 
                                   (self.point_positions[:, 1] - y_click)**2)
                
                for idx in self.deleted_indices:
                    if idx < len(distances):
                        distances[idx] = np.inf
                
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < 10:
                    self.pointClicked.emit(int(nearest_idx))
                    return
        
        super().mousePressEvent(event)

# ============================================================================
# UNIFIED PIPELINE WORKER 
# ============================================================================

class PipelineWorker(QThread):
    """Worker thread for running pipeline steps - completely step-agnostic"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    session_progress = pyqtSignal(int, int, float)  # For Step 1
    animal_progress = pyqtSignal(int, int, float)   # For other steps
    
    def __init__(self, step, config, loaded_sessions=None):
        super().__init__()
        self.step = step
        self.config = config
        self.loaded_sessions = loaded_sessions
        self._step_result = None
        self._step_error = None
        
    def run(self):
        """Run the pipeline step using unified monitoring"""
        try:
            step_info = get_step_info(self.step)
            if not step_info:
                self.error.emit(f"Unknown step: {self.step}")
                return
            
            self.progress.emit(f"Starting {step_info['label']}...")
            
            # Setup logging
            gui_handler = self._setup_logging(step_info['logger_name'])
            
            try:
                self._run_with_monitoring(step_info, gui_handler)
            finally:
                self._cleanup_logging(step_info['logger_name'], gui_handler)
                
        except Exception as e:
            import traceback
            error_msg = f"Error in Step {self.step}:\n{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _setup_logging(self, logger_name):
        """Setup logging handler for step"""
        class GUIHandler(logging.Handler):
            def __init__(self, signal):
                super().__init__()
                self.signal = signal
            def emit(self, record):
                try:
                    self.signal.emit(self.format(record))
                except:
                    pass
        
        gui_handler = GUIHandler(self.progress)
        gui_handler.setFormatter(logging.Formatter('%(message)s'))
        
        logger = logging.getLogger(logger_name)
        logger.addHandler(gui_handler)
        logger.setLevel(logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)
        root_logger.setLevel(logging.INFO)
        
        return gui_handler
    
    def _cleanup_logging(self, logger_name, gui_handler):
        """Cleanup logging handlers"""
        logger = logging.getLogger(logger_name)
        logger.removeHandler(gui_handler)
        root_logger = logging.getLogger()
        root_logger.removeHandler(gui_handler)
    
    def _run_with_monitoring(self, step_info, gui_handler):
        """Unified monitoring for all steps"""
        # Get total items using centralized function
        total_items = count_items_for_step(self.step, self.config.output_dir, self.loaded_sessions)
        self.progress.emit(f"Found {total_items} {step_info['count_unit']} to process")
        
        # Setup output directory
        output_dir = get_step_output_dir(self.step, self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file pattern
        file_pattern = get_step_file_pattern(self.step)
        
        # Track existing files
        initial_mod_times = {}
        for f in output_dir.glob(file_pattern):
            try:
                initial_mod_times[f.name] = f.stat().st_mtime
            except:
                pass
        
        # Start processing in background
        processing_complete = threading.Event()
        
        def run_processing():
            try:
                # Dynamically import and run using metadata
                module = importlib.import_module(f"steps.{step_info['run_module']}")
                run_func = getattr(module, step_info['run_function'])
                
                # Build kwargs
                kwargs = build_run_kwargs(self.step, self.config)
                
                # Special handling for Step 1 which needs sessions + callback
                run_kwargs_info = step_info['run_kwargs']
                if run_kwargs_info.get('needs_sessions') and run_kwargs_info.get('needs_callback'):
                    self._step_result = run_func(
                        self.config,
                        loaded_sessions=self.loaded_sessions,
                        session_callback=self._emit_session_progress
                    )
                else:
                    self._step_result = run_func(**kwargs)
                    
            except Exception as e:
                self._step_error = str(e)
                import traceback
                traceback.print_exc()
            finally:
                processing_complete.set()
        
        thread = threading.Thread(target=run_processing)
        thread.daemon = True
        thread.start()
        
        # Monitor files
        self._monitor_files(
            output_dir, 
            file_pattern,
            initial_mod_times,
            total_items,
            thread,
            processing_complete,
            step_info['progress_signal']
        )
        
        thread.join(timeout=5.0)
        
        # Emit results
        if self._step_error:
            self.error.emit(self._step_error)
        elif self._step_result is not None:
            self.finished.emit(self._step_result)
        else:
            self.error.emit(f"Step {self.step} completed but no result was generated")
    
    def _monitor_files(self, output_dir, file_pattern, initial_mod_times, 
                      total_items, thread, processing_complete, progress_signal):
        """Monitor file creation and emit progress"""
        completed_items = 0
        item_times = []
        last_count = 0
        start_time = time.time()
        last_completion_time = start_time
        max_wait_cycles = 0
        
        while thread.is_alive() or (not processing_complete.is_set() and max_wait_cycles < 10):
            current_files = list(output_dir.glob(file_pattern))
            
            updated_files = []
            for f in current_files:
                try:
                    current_mtime = f.stat().st_mtime
                    if f.name not in initial_mod_times or current_mtime > initial_mod_times[f.name]:
                        updated_files.append(f)
                except (FileNotFoundError, OSError):
                    continue
            
            completed_item_ids = set()
            for f in updated_files:
                try:
                    item_id = f.stem.split('_')[0]
                    completed_item_ids.add(item_id)
                except:
                    pass
            
            completed_items = len(completed_item_ids)
            
            if completed_items > last_count:
                current_time = time.time()
                time_since_last = current_time - last_completion_time
                item_times.append(time_since_last)
                last_completion_time = current_time
                last_count = completed_items
                max_wait_cycles = 0
                
                avg_time = np.mean(item_times) if item_times else time_since_last
                
                # Emit appropriate progress signal based on metadata
                if progress_signal == 'animal_progress':
                    self.animal_progress.emit(completed_items, total_items, avg_time)
                # session_progress is emitted directly from callback
            
            if not thread.is_alive() and completed_items < total_items:
                max_wait_cycles += 1
            
            time.sleep(0.5)
    
    def _emit_session_progress(self, current, total, session_time):
        """Helper to emit session progress"""
        self.session_progress.emit(current, total, session_time)

# ============================================================================
# PIPELINE CONFIG DIALOG - FULLY SCHEMA DRIVEN
# ============================================================================

class PipelineConfigDialog(QDialog):
    """Dialog for configuring pipeline parameters - 100% SCHEMA DRIVEN"""
    
    def __init__(self, config, parent=None):
        super().__init__()
        self.config = config
        self.param_widgets = {}
        from utilities import apply_dark_theme_to_widget
        apply_dark_theme_to_widget(self)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI dynamically from PARAMETER_SCHEMAS"""
        self.setWindowTitle('Pipeline Configuration')
        self.setMinimumWidth(700)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel('⚙️ Pipeline Configuration')
        title.setStyleSheet('font-size: 16px; font-weight: bold; padding: 10px; color: white;')
        layout.addWidget(title)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(500)
        scroll.setStyleSheet('''
            QScrollArea {
                background-color: #1E1E1E;
                border: 1px solid #00AAFF;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background-color: #1E1E1E;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #00AAFF;
                border-radius: 6px;
                min-height: 20px;
            }
        ''')
        
        params_widget = QWidget()
        params_widget.setStyleSheet("background-color: #1E1E1E;")
        params_layout = QVBoxLayout(params_widget)
        
        # Directories group
        self._add_directories_group(params_layout)
        
        # General parameters group
        self._add_parameter_group(params_layout, "General", [
            'n_workers', 'verbose', 'skip_existing'
        ])
        
        # Step 1 parameters
        self._add_parameter_group(params_layout, "Step 1: Quad Generation", [
            'min_tri_area', 'height_percentile', 'min_third_points_per_diagonal',
            'triangle_chunk_size', 'max_triangles_per_diagonal', 'quad_keep_fraction',
            'min_pairwise_distance'
        ])
        
        # Step 1.5 parameters
        self._add_parameter_group(params_layout, "Step 1.5: Calibration", [
            'sample_size', 'target_quality', 'threshold_min', 'threshold_max', 'n_threshold_points'
        ])
        
        # Step 2 parameters
        self._add_parameter_group(params_layout, "Step 2: Quad Matching", [
            'threshold', 'distance_metric', 'consistency_threshold'
        ])
        
        # Step 2.5 parameters
        self._add_parameter_group(params_layout, "Step 2.5: RANSAC", [
            'ransac_max_residual', 'ransac_iterations', 'ransac_min_inlier_ratio'
        ])
        
        # Step 3 parameters
        self._add_parameter_group(params_layout, "Step 3: Final Matching", [
            'target_match_rate', 'use_quad_voting', 'hungarian_max_cost'
        ])
        
        scroll.setWidget(params_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton('🔄 Reset All to Defaults')
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setStyleSheet('''
            QPushButton {
                background-color: #555555;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        ''')
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton('❌ Cancel')
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet('''
            QPushButton {
                background-color: #555;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        ''')
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton('💾 Save Configuration')
        save_btn.clicked.connect(self.accept_and_apply)
        save_btn.setStyleSheet('''
            QPushButton {
                background-color: #00CC00;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00DD00;
            }
        ''')
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _add_directories_group(self, parent_layout):
        """Add input/output directory controls"""
        group = QGroupBox("📁 Directories")
        group.setStyleSheet('''
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2A2A2A;
            }
        ''')
        
        layout = QVBoxLayout()
        
        # Input directory (read-only)
        input_row = QHBoxLayout()
        input_label = QLabel("Input Directory:")
        input_label.setStyleSheet("color: white; font-size: 12px;")
        input_label.setMinimumWidth(150)
        input_row.addWidget(input_label)
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setText(self.config.input_dir if self.config.input_dir else "Not set")
        self.input_dir_edit.setReadOnly(True)
        self.input_dir_edit.setStyleSheet("background-color: #1A1A1A; color: #888; border: 1px solid #555; padding: 3px;")
        input_row.addWidget(self.input_dir_edit)
        
        layout.addLayout(input_row)
        
        # Output directory (editable)
        output_row = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        output_label.setStyleSheet("color: white; font-size: 12px;")
        output_label.setMinimumWidth(150)
        output_row.addWidget(output_label)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.config.output_dir if self.config.output_dir else "")
        self.output_dir_edit.setStyleSheet("background-color: #1A1A1A; color: white; border: 1px solid #555; padding: 3px;")
        output_row.addWidget(self.output_dir_edit)
        
        browse_btn = QPushButton('📁 Browse')
        browse_btn.clicked.connect(self.browse_output_dir)
        browse_btn.setMaximumWidth(100)
        browse_btn.setStyleSheet('''
            QPushButton {
                background-color: #4400AA;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5500BB;
            }
        ''')
        output_row.addWidget(browse_btn)
        
        layout.addLayout(output_row)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _add_parameter_group(self, parent_layout, title, param_names):
        """Add a group of parameters from schemas"""
        group = QGroupBox(f"⚙️ {title}")
        group.setStyleSheet('''
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2A2A2A;
            }
            QLabel {
                background-color: transparent;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1A1A1A;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
            QCheckBox {
                color: white;
            }
        ''')
        
        layout = QVBoxLayout()
        
        for param_name in param_names:
            self._add_schema_param(layout, param_name)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _add_schema_param(self, layout, param_name):
        """Add a parameter widget from PARAMETER_SCHEMAS"""
        schema = get_parameter_schema(param_name)
        if not schema:
            return
        
        # Get current value from config
        value = getattr(self.config, param_name, schema['default'])
        
        row = QHBoxLayout()
        
        # Label
        label_text = schema.get('description', param_name.replace('_', ' ').title())
        label_widget = QLabel(f"{label_text}:")
        label_widget.setStyleSheet("color: white; font-size: 12px;")
        label_widget.setMinimumWidth(250)
        row.addWidget(label_widget)
        
        # Widget based on schema
        widget_type = schema.get('widget', 'lineedit')
        
        if widget_type == 'spinbox':
            widget = QSpinBox()
            widget.setRange(schema.get('min', 0), schema.get('max', 999999))
            widget.setValue(int(value) if value is not None else schema['default'])
            
        elif widget_type == 'doublespinbox':
            widget = QDoubleSpinBox()
            widget.setRange(schema.get('min', 0.0), schema.get('max', 999999.0))
            widget.setDecimals(schema.get('decimals', 2))
            widget.setSingleStep(schema.get('step', 0.1))
            
            if schema.get('nullable') and value is None:
                widget.setValue(schema.get('min', 0.0))
                widget.setSpecialValueText("Auto")
            else:
                widget.setValue(float(value) if value is not None else schema['default'])
                
        elif widget_type == 'checkbox':
            widget = QCheckBox()
            widget.setChecked(bool(value) if value is not None else schema['default'])
            
        elif widget_type == 'combobox':
            widget = QComboBox()
            options = schema.get('options', [])
            widget.addItems(options)
            if value in options:
                widget.setCurrentText(value)
            
        else:
            widget = QLineEdit(str(value) if value is not None else str(schema['default']))
        
        row.addWidget(widget)
        self.param_widgets[param_name] = widget
        
        layout.addLayout(row)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        current_dir = self.output_dir_edit.text()
        if not current_dir:
            current_dir = self.config.input_dir if self.config.input_dir else "."
        
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", current_dir)
        if folder:
            self.output_dir_edit.setText(folder)
    
    def reset_to_defaults(self):
        """Reset all parameters to defaults from PARAMETER_SCHEMAS"""
        for param_name, widget in self.param_widgets.items():
            schema = get_parameter_schema(param_name)
            if not schema:
                continue
            
            default = schema['default']
            
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default) if default is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                if default is None and schema.get('nullable'):
                    widget.setValue(widget.minimum())
                else:
                    widget.setValue(float(default) if default is not None else 0.0)
            elif isinstance(widget, QComboBox):
                if default in schema.get('options', []):
                    widget.setCurrentText(default)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(default) if default is not None else "")
        
        QMessageBox.information(self, 'Reset', 'All parameters reset to defaults from step_info.py')
    
    def accept_and_apply(self):
        """Apply changes to config and accept"""
        # Apply all parameter widgets
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                setattr(self.config, param_name, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                setattr(self.config, param_name, widget.text())
            elif isinstance(widget, QSpinBox):
                setattr(self.config, param_name, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                schema = get_parameter_schema(param_name)
                if schema and schema.get('nullable') and widget.value() == widget.minimum():
                    setattr(self.config, param_name, None)
                else:
                    setattr(self.config, param_name, widget.value())
            elif isinstance(widget, QComboBox):
                setattr(self.config, param_name, widget.currentText())
        
        # Update output directory
        new_output = self.output_dir_edit.text()
        if new_output and new_output != self.config.output_dir:
            self.config.output_dir = new_output
            self.config.output_path = Path(new_output)
            self.config.intermediate_path = Path(new_output) / "intermediate"
        
        self.accept()
    
    def get_config(self):
        """Get updated configuration"""
        return self.config

# ============================================================================
# REST OF COMPONENTS 
# ============================================================================

class StepConfirmationDialog(QDialog):
    """Step-specific confirmation dialog - uses step_info.py for metadata"""
    
    def __init__(self, step, config, parent=None):
        super().__init__()
        self.step = step
        self.config = config
        self.param_widgets = {}
        from utilities import apply_dark_theme_to_widget
        apply_dark_theme_to_widget(self)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        step_info = get_step_info(self.step)
        step_name = step_info.get('name', f'Step {self.step}')
        step_icon = step_info.get('icon', '▶️')
        
        self.setWindowTitle(f'Confirm {step_info.get("label", f"Step {self.step}")}')
        self.setMinimumWidth(600)
        
        layout = QVBoxLayout()
        
        title = QLabel(f'{step_icon} Run {step_info.get("label", f"Step {self.step}")}')
        title.setStyleSheet('font-size: 16px; font-weight: bold; padding: 10px; color: white;')
        layout.addWidget(title)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)
        scroll.setMaximumHeight(500)
        scroll.setStyleSheet('''
            QScrollArea {
                background-color: #1E1E1E;
                border: 1px solid #00AAFF;
                border-radius: 5px;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #1E1E1E;
            }
            QScrollBar:vertical {
                background-color: #1E1E1E;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #00AAFF;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00DDFF;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        ''')

        params_widget = QWidget()
        params_widget.setStyleSheet("background-color: #1E1E1E;")
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(5, 5, 5, 5)

        self._add_step_parameters(params_layout, step_info)

        scroll.setWidget(params_widget)
        layout.addWidget(scroll)
                
        message = QLabel(self._get_step_message(step_info))
        message.setWordWrap(True)
        message.setStyleSheet('padding: 10px; color: #FFD700; font-size: 12px;')
        layout.addWidget(message)
        
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton('🔄 Reset to Defaults')
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setStyleSheet('''
            QPushButton {
                background-color: #555555;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        ''')
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton('❌ Cancel')
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet('''
            QPushButton {
                background-color: #555;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        ''')
        button_layout.addWidget(cancel_btn)
        
        run_btn = QPushButton('▶️  Run Step')
        run_btn.clicked.connect(self.accept_and_apply)
        run_btn.setStyleSheet('''
            QPushButton {
                background-color: #00CC00;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #00DD00;
            }
        ''')
        button_layout.addWidget(run_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _add_schema_param(self, layout, param_name):
        """Add a parameter row using PARAMETER_SCHEMAS"""
        schema = get_parameter_schema(param_name)
        if not schema:
            return
        
        value = getattr(self.config, param_name, schema['default'])
        
        row = QHBoxLayout()
        
        label_text = schema.get('description', param_name.replace('_', ' ').title())
        label_widget = QLabel(f"{label_text}:")
        label_widget.setStyleSheet("color: white; font-size: 12px; background-color: transparent;")
        label_widget.setMinimumWidth(200)
        row.addWidget(label_widget)
        
        widget_type = schema.get('widget', 'lineedit')
        
        if widget_type == 'spinbox':
            widget = QSpinBox()
            widget.setRange(schema.get('min', 0), schema.get('max', 999999))
            widget.setValue(int(value) if value is not None else schema['default'])
            
        elif widget_type == 'doublespinbox':
            widget = QDoubleSpinBox()
            widget.setRange(schema.get('min', 0.0), schema.get('max', 999999.0))
            widget.setDecimals(schema.get('decimals', 2))
            widget.setSingleStep(schema.get('step', 0.1))
            
            if schema.get('nullable') and value is None:
                widget.setValue(schema.get('min', 0.0))
                widget.setSpecialValueText("Auto")
            else:
                widget.setValue(float(value) if value is not None else schema['default'])
                
        elif widget_type == 'checkbox':
            widget = QCheckBox()
            widget.setChecked(bool(value) if value is not None else schema['default'])
            
        elif widget_type == 'combobox':
            widget = QComboBox()
            options = schema.get('options', [])
            widget.addItems(options)
            if value in options:
                widget.setCurrentText(value)
            
        else:
            widget = QLineEdit(str(value) if value is not None else str(schema['default']))
        
        row.addWidget(widget)
        self.param_widgets[param_name] = widget
        
        layout.addLayout(row)

    def _add_step_parameters(self, layout, step_info):
        """Add step-specific editable parameters using schemas from step_info"""
        
        # Common parameters group
        common_group = self._create_param_group("📁 Common Parameters")
        common_layout = QVBoxLayout()
        
        self._add_param_row(common_layout, "Output Directory", "output_dir", 
                        self.config.output_dir, "readonly")
        self._add_schema_param(common_layout, "n_workers")
        
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)
        
        # Step-specific parameters - pull from step_info
        step_params = get_step_parameters(self.step)
        
        if step_params:
            step_group = self._create_param_group(f"⚙️ {step_info['name']} Parameters")
            step_layout = QVBoxLayout()
            
            for param_name in step_params:
                self._add_schema_param(step_layout, param_name)
            
            step_group.setLayout(step_layout)
            layout.addWidget(step_group)
            
    def _create_param_group(self, title):
        """Create a styled parameter group"""
        group = QGroupBox(title)
        group.setStyleSheet('''
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #00AAFF;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: white;
                background-color: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2A2A2A;
            }
            QLabel {
                background-color: transparent;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #1A1A1A;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #00AAFF;
            }
            QLineEdit {
                background-color: #1A1A1A;
                color: #888;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
        ''')
        return group

    def _add_param_row(self, layout, label, param_name, value, widget_type, 
                    min_val=None, max_val=None, step=None, decimals=2, tooltip=None):
        """Add a parameter row with label and editable widget"""
        row = QHBoxLayout()
        
        label_widget = QLabel(f"{label}:")
        label_widget.setStyleSheet("color: white; font-size: 12px; background-color: transparent;")
        label_widget.setMinimumWidth(200)
        if tooltip:
            label_widget.setToolTip(tooltip)
        row.addWidget(label_widget)
        
        if widget_type == "readonly":
            widget = QLineEdit(str(value))
            widget.setReadOnly(True)
        elif widget_type == "spinbox":
            widget = QSpinBox()
            widget.setRange(min_val or 0, max_val or 999999)
            widget.setValue(int(value))
        elif widget_type == "doublespinbox":
            widget = QDoubleSpinBox()
            widget.setRange(min_val or 0.0, max_val or 999999.0)
            widget.setSingleStep(step or 0.1)
            widget.setDecimals(decimals)
            widget.setValue(float(value))
        elif widget_type == "checkbox":
            widget = QCheckBox()
            widget.setChecked(bool(value))
        else:
            widget = QLineEdit(str(value))
        
        if tooltip:
            widget.setToolTip(tooltip)
        
        row.addWidget(widget)
        self.param_widgets[param_name] = widget
        
        layout.addLayout(row)

    def reset_to_defaults(self):
        """Reset all parameters to defaults from PARAMETER_SCHEMAS"""
        for param_name, widget in self.param_widgets.items():
            schema = get_parameter_schema(param_name)
            if not schema:
                continue
                
            default = schema['default']
            
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default) if default is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                if default is None and schema.get('nullable'):
                    widget.setValue(widget.minimum())
                else:
                    widget.setValue(float(default) if default is not None else 0.0)
            elif isinstance(widget, QComboBox):
                if default in schema.get('options', []):
                    widget.setCurrentText(default)
            elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
                widget.setText(str(default) if default is not None else "")
        
        QMessageBox.information(self, 'Reset', 'Parameters reset to defaults from step_info.py')


    def accept_and_apply(self):
        """Apply changes to config and accept dialog"""
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                setattr(self.config, param_name, widget.isChecked())
            elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
                setattr(self.config, param_name, widget.text())
            elif isinstance(widget, QSpinBox):
                setattr(self.config, param_name, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                setattr(self.config, param_name, widget.value())
        
        self.accept()
    
    def _get_step_message(self, step_info) -> str:
        """Get step-specific warning/info message"""
        prereqs = step_info.get('prerequisites', [])
        prereq_text = f"Requires Step {', '.join(map(str, prereqs))} results" if prereqs else "No prerequisites"
        
        return f"⚠️  {step_info.get('description', '')}\n{prereq_text}"

def open_results_inspector(step, config, parent=None):
    """Open the appropriate results inspector for a pipeline step"""
    from viewers import (
        Step1Viewer, Step1_5Viewer, Step2Viewer, Step2_5Viewer, Step3Viewer,
    )
    
    # Parse step if it's a directory name string
    if isinstance(step, str):
        from .step_info import parse_step_from_dirname
        parsed = parse_step_from_dirname(step)
        if parsed:
            step = parsed
        else:
            QMessageBox.critical(parent, 'Error', f'Unknown step: {step}')
            return
    
    # Map steps to viewers
    viewers = {
        1: Step1Viewer,
        1.5: Step1_5Viewer,
        2: Step2Viewer,
        2.5: Step2_5Viewer,
        3: Step3Viewer,
    }
    
    try:
        viewer_class = viewers.get(step)
        if viewer_class:
            viewer = viewer_class(config, parent)
            viewer.exec_()
        else:
            QMessageBox.critical(parent, 'Error', f'No viewer for step {step}')
    except ImportError as e:
        QMessageBox.warning(parent, 'Import Error', f'Could not import viewer:\n{e}')
    except Exception as e:
        import traceback
        QMessageBox.critical(parent, 'Error', f'Error launching viewer:\n{e}\n\n{traceback.format_exc()}')

def create_progress_dialog_with_log(parent, title, step, maximum=0):
    """Create a progress dialog with embedded log window"""
    step_info = get_step_info(step)
    label = step_info.get('label', f'Step {step}') if step_info else f'Step {step}'
    
    progress = QProgressDialog(f'Running {label}...', 'Cancel', 0, maximum, parent)
    progress.setWindowModality(Qt.WindowModal)
    progress.setWindowTitle(title)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    
    progress.start_time = time.time()
    progress.session_times = []
    
    log_window = QDialog(parent)
    log_window.setWindowTitle(f'{label} Log')
    log_window.resize(600, 400)
    log_layout = QVBoxLayout()
    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setFont(QFont('Courier', 9))
    log_layout.addWidget(log_text)
    log_window.setLayout(log_layout)
    log_window.show()
    
    return progress, log_window, log_text

def apply_button_style(button, style_type='primary'):
    """Apply consistent button styling"""
    styles = {
        'primary': '''
            QPushButton {
                background-color: #00AAFF;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0088DD;
            }
        ''',
        'pipeline': '''
            QPushButton {
                background-color: #4400AA;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5500BB;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        ''',
        'success': '''
            QPushButton {
                background-color: #00CC00;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00DD00;
            }
        ''',
        'config': '''
            QPushButton {
                background-color: #6600CC;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7700DD;
            }
        '''
    }
    
    button.setStyleSheet(styles.get(style_type, styles['primary']))

def create_pipeline_callbacks(parent_viewer, step, config):
    """Create standardized callbacks for pipeline step execution"""
    from PyQt5.QtWidgets import QMessageBox, QApplication
    from .step_info import get_step_enables
    
    def on_progress(msg):
        """Handle progress updates from worker thread"""
        if hasattr(parent_viewer, '_current_log_text'):
            parent_viewer._current_log_text.append(msg)
            parent_viewer._current_log_text.verticalScrollBar().setValue(
                parent_viewer._current_log_text.verticalScrollBar().maximum()
            )
        print(msg)
        QApplication.processEvents()
    
    def on_finished(result):
        """Handle successful completion"""
        if hasattr(parent_viewer, '_current_progress'):
            parent_viewer._current_progress.close()
        
        # Enable next steps based on metadata
        step_key = str(step).replace('.', '_')
        inspect_btn_name = f'step{step_key}_inspect_btn'
        if hasattr(parent_viewer, inspect_btn_name):
            getattr(parent_viewer, inspect_btn_name).setEnabled(True)
        
        # Enable steps that this step enables
        for enabled_step in get_step_enables(step):
            enabled_key = str(enabled_step).replace('.', '_')
            btn_name = f'step{enabled_key}_btn'
            if hasattr(parent_viewer, btn_name):
                getattr(parent_viewer, btn_name).setEnabled(True)
        
        step_info = get_step_info(step)
        label = step_info.get('label', f'Step {step}') if step_info else f'Step {step}'
        
        QMessageBox.information(
            parent_viewer, 'Success', 
            f'{label} completed successfully!\n\nResults saved to:\n{config.output_dir}'
        )
    
    def on_error(error_msg):
        """Handle errors"""
        if hasattr(parent_viewer, '_current_progress'):
            parent_viewer._current_progress.close()
        
        import traceback
        if hasattr(parent_viewer, '_current_log_text'):
            parent_viewer._current_log_text.append(f'\n{"="*60}')
            parent_viewer._current_log_text.append(f'ERROR: {error_msg}')
            parent_viewer._current_log_text.append(f'{"="*60}')
            parent_viewer._current_log_text.append('\nTraceback:')
            parent_viewer._current_log_text.append(traceback.format_exc())
        
        print(f'\nERROR: {error_msg}')
        print(traceback.format_exc())
        
        step_info = get_step_info(step)
        label = step_info.get('label', f'Step {step}') if step_info else f'Step {step}'
        
        QMessageBox.critical(parent_viewer, 'Error', f'{label} failed:\n{error_msg}')
    
    return on_progress, on_finished, on_error

def confirm_step_execution(step, config, parent=None) -> bool:
    """Show confirmation dialog before running a pipeline step"""
    dialog = StepConfirmationDialog(step, config, parent)
    return dialog.exec_() == QDialog.Accepted
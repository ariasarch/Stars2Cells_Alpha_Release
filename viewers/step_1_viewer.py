"""
Step 1 Results Viewer: Quad Generation Inspector 
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QSpinBox, QMessageBox, QLabel,
                             QSplitter, QTabWidget, QWidget, QPushButton)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from utilities import *

class Step1Viewer(QDialog):
    """Viewer for Step 1: Quad Generation Results"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_session = None
        self.session_data = {}
        self.animal_sessions = {}
        
        self.setWindowTitle('Step 1: Quad Generation Results')
        self.resize(1400, 900)
        
        self.init_ui()
        self.load_results()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Top controls using utility
        self.controls = create_top_controls(
            'Session Selection',
            combos=[
                ('Animal', self.on_animal_changed),
                ('Session', self.on_session_changed),
            ],
            refresh_callback=self.load_results
        )
        self.animal_combo, self.session_combo = self.controls.combos
        layout.addWidget(self.controls)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Data table and stats
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right: Visualizations
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Bottom: Action buttons using utility
        buttons = create_action_buttons({
            '💾 Export Summary': self.export_summary,
            '✖ Close': self.accept
        })
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def create_left_panel(self):
        """Create left panel with stats and data table"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary stats using utility
        stat_labels = [
            ('n_rois', 'ROIs'),
            ('n_quads', 'Quads Generated'),
            ('avg_area', 'Avg Quad Area'),
            ('min_area', 'Min Quad Area'),
            ('max_area', 'Max Quad Area'),
        ]
        self.stats_group, self.stat_labels = create_stat_group(
            'Session Statistics', stat_labels
        )
        layout.addWidget(self.stats_group)
        
        # Quad data table
        table_group = QGroupBox('Quad Details')
        table_layout = QVBoxLayout()
        
        self.quad_table = QTableWidget()
        self.quad_table.setColumnCount(7)
        self.quad_table.setHorizontalHeaderLabels([
            'Quad ID', 'Point 1', 'Point 2', 'Point 3', 'Point 4', 
            'Area', 'Aspect Ratio'
        ])
        table_layout.addWidget(self.quad_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        return widget
        
    def create_right_panel(self):
        """Create right panel with visualizations"""
        tabs = QTabWidget()
        
        # Tab 1: Spatial plot using utility
        spatial_tab, self.spatial_plot = create_pyqtgraph_tab('Spatial View')
        setup_pyqtgraph_plot(
            self.spatial_plot, 
            'ROIs and Quads',
            'X Coordinate (pixels)', 
            'Y Coordinate (pixels)'
        )
        tabs.addTab(spatial_tab, '📍 Spatial View')
        
        # Tab 2: Area distribution using utility
        area_tab, self.area_figure, self.area_canvas = create_matplotlib_tab(
            'Area Distribution'
        )
        tabs.addTab(area_tab, '📊 Area Distribution')
        
        # Tab 3: Aspect ratio distribution using utility
        aspect_tab, self.aspect_figure, self.aspect_canvas = create_matplotlib_tab(
            'Aspect Ratios'
        )
        tabs.addTab(aspect_tab, '📐 Aspect Ratios')
        
        # Tab 4: Spatial participation (NEW - same as Step 2)
        participation_tab, self.participation_figure, self.participation_canvas = create_matplotlib_tab('Spatial Participation')
        tabs.addTab(participation_tab, '🎯 Spatial Participation')
        
        # Tab 5: Configuration
        config_tab = self.create_config_tab()
        tabs.addTab(config_tab, '⚙️ Parameters')
        
        # Tab 6: Statistics
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(1, self.config.output_dir)
        tabs.addTab(stats_tab, '📋 Statistics')
        
        return tabs
        
    def create_config_tab(self):
        """Create configuration tab for parameter adjustment"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info = QLabel('Adjust parameters and re-run quad generation for selected session')
        info.setStyleSheet('font-weight: bold; padding: 10px;')
        layout.addWidget(info)
        
        # Parameter form
        form = QFormLayout()
        
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.1, 1000.0)
        self.min_area_spin.setValue(self.config.min_tri_area)
        self.min_area_spin.setDecimals(2)
        form.addRow('Min Triangle Area:', self.min_area_spin)
        
        self.height_percentile_spin = QDoubleSpinBox()
        self.height_percentile_spin.setRange(0.0, 100.0)
        self.height_percentile_spin.setValue(self.config.height_percentile)
        self.height_percentile_spin.setDecimals(1)
        form.addRow('Height Percentile:', self.height_percentile_spin)
        
        self.min_third_spin = QSpinBox()
        self.min_third_spin.setRange(1, 100)
        self.min_third_spin.setValue(self.config.min_third_points_per_diagonal)
        form.addRow('Min Third Points:', self.min_third_spin)
        
        layout.addLayout(form)
        
        # Action buttons using utility
        buttons = create_action_buttons({
            '↺ Reset to Default': self.reset_params,
            '▶️ Re-run Current Session': self.rerun_current_session,
            '▶️ Re-run All Sessions': self.rerun_all_sessions,
        }, add_stretch=False)
        layout.addWidget(buttons)
        
        layout.addStretch()
        
        return widget
        
    def load_results(self):
        """Load Step 1 results from step_1_results directory"""
        # Use step_1_results instead of intermediate
        step1_dir = get_step_results_dir(self.config.output_dir, 1)
        
        print(f"\n{'='*60}")
        print(f"Loading Step 1 results from: {step1_dir}")
        
        if not step1_dir.exists():
            show_no_results_error(self, step1_dir, 'Step 1')
            return
        
        # Use correct file pattern
        quad_files = scan_results_directory(
            step1_dir, 
            get_step_file_pattern(1),  # '*_centroids_quads.npz'
            verbose=True
        )
        
        if not quad_files:
            show_no_files_error(self, step1_dir, '*_centroids_quads.npz', 'Step 1')
            return
        
        print(f"Found {len(quad_files)} quad files")
        
        # Clear previous data
        self.session_data.clear()
        self.animal_sessions.clear()
        self.animal_combo.clear()
        self.session_combo.clear()
        
        # Load each file
        loaded_count = 0
        for quad_file in quad_files:
            # Load using utility
            data = load_npz_safely(quad_file, verbose=True)
            if data is None:
                continue
            
            # Extract animal and session using utility
            result = extract_animal_session_from_filename(
                quad_file.stem, 
                suffix='_centroids_quads'
            )
            
            if result:
                animal_id, session_id = result
                
                if animal_id not in self.animal_sessions:
                    self.animal_sessions[animal_id] = []
                
                self.animal_sessions[animal_id].append(session_id)
                
                session_key = f"{animal_id}_{session_id}"
                self.session_data[session_key] = {
                    'file': quad_file,
                    'data': data,
                    'animal': animal_id,
                    'session': session_id
                }
                
                loaded_count += 1
                print(f"  ✓ Loaded as {session_key}")
        
        print(f"{'='*60}")
        print(f"Successfully loaded {loaded_count} sessions")
        print(f"{'='*60}\n")
        
        if loaded_count == 0:
            QMessageBox.warning(
                self, 'Load Failed',
                f'Could not load any quad files from:\n{step1_dir}'
            )
            return
        
        # Populate animal combo
        animals = sorted(self.animal_sessions.keys())
        self.animal_combo.addItems(animals)
        
        if animals:
            self.animal_combo.setCurrentIndex(0)
        
        self.load_pipeline_stats()

    def on_animal_changed(self, animal_id):
        """Handle animal selection change"""
        if not animal_id or animal_id not in self.animal_sessions:
            return
        
        self.current_animal = animal_id
        
        # Update session combo
        self.session_combo.clear()
        sessions = sorted(self.animal_sessions[animal_id])
        self.session_combo.addItems(sessions)
        
        if sessions:
            self.session_combo.setCurrentIndex(0)
    
    def on_session_changed(self, session_id):
        """Handle session selection change"""
        if not session_id or not self.current_animal:
            return
        
        self.current_session = session_id
        session_key = f"{self.current_animal}_{session_id}"
        
        if session_key in self.session_data:
            self.display_session(session_key)
    
    def display_session(self, session_key):
        """Display data for selected session"""
        session_info = self.session_data[session_key]
        data = session_info['data']
        
        print(f"\nDisplaying session: {session_key}")
        
        # Extract quad data
        quad_idx = data.get('quad_idx', np.array([]))
        centroids = data['centroids']  # (N, 2) as (y, x)
        
        # Calculate quad areas
        quad_areas = self._calculate_quad_areas(centroids, quad_idx)
        
        # Update statistics using utility
        n_rois = len(centroids)
        n_quads = len(quad_idx)
        
        stats = {
            'n_rois': n_rois,
            'n_quads': n_quads,
        }
        
        if len(quad_areas) > 0:
            stats['avg_area'] = np.mean(quad_areas)
            stats['min_area'] = np.min(quad_areas)
            stats['max_area'] = np.max(quad_areas)
        
        update_stat_labels(self.stat_labels, stats)
        
        # Update quad table
        self._update_quad_table(centroids, quad_idx, quad_areas)
        
        # Update plots
        centroids_x = centroids[:, 1]
        centroids_y = centroids[:, 0]
        
        self.plot_spatial(centroids_x, centroids_y, quad_idx)
        self.plot_area_distribution(quad_areas)
        self.plot_aspect_ratios(centroids_x, centroids_y, quad_idx)
        self.plot_spatial_participation(centroids_x, centroids_y, quad_idx)
        
        print(f"Display complete\n")
    
    def _calculate_quad_areas(self, centroids, quad_idx):
        """Calculate areas for all quads"""
        quad_areas = []
        for quad in quad_idx:
            if len(quad) == 4:
                pts = centroids[quad]  # (4, 2) in (y, x)
                x = pts[:, 1]
                y = pts[:, 0]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                quad_areas.append(area)
            else:
                quad_areas.append(0)
        return np.array(quad_areas)
    
    def _update_quad_table(self, centroids, quad_idx, quad_areas):
        """Update the quad details table"""
        n_quads = len(quad_idx)
        n_display = min(n_quads, 1000)
        
        self.quad_table.setRowCount(n_display)
        
        centroids_x = centroids[:, 1]
        centroids_y = centroids[:, 0]
        
        for i in range(n_display):
            quad = quad_idx[i]
            area = quad_areas[i] if i < len(quad_areas) else 0
            
            # Calculate aspect ratio
            if len(quad) == 4:
                pts = np.array([[centroids_x[quad[j]], centroids_y[quad[j]]] for j in range(4)])
                d1 = np.linalg.norm(pts[0] - pts[2])
                d2 = np.linalg.norm(pts[1] - pts[3])
                aspect = max(d1, d2) / (min(d1, d2) + 1e-6)
            else:
                aspect = 0
            
            self.quad_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.quad_table.setItem(i, 1, QTableWidgetItem(str(quad[0]) if len(quad) > 0 else '--'))
            self.quad_table.setItem(i, 2, QTableWidgetItem(str(quad[1]) if len(quad) > 1 else '--'))
            self.quad_table.setItem(i, 3, QTableWidgetItem(str(quad[2]) if len(quad) > 2 else '--'))
            self.quad_table.setItem(i, 4, QTableWidgetItem(str(quad[3]) if len(quad) > 3 else '--'))
            self.quad_table.setItem(i, 5, QTableWidgetItem(f"{area:.2f}"))
            self.quad_table.setItem(i, 6, QTableWidgetItem(f"{aspect:.2f}"))
    
    def plot_spatial(self, x, y, quads):
        """Plot spatial distribution of ROIs and quads"""
        self.spatial_plot.clear()
        
        # Plot all ROIs
        scatter = pg.ScatterPlotItem(
            x, y,
            size=8,
            brush=pg.mkBrush(0, 100, 255, 120),
            pen=pg.mkPen('w', width=1)
        )
        self.spatial_plot.addItem(scatter)
        
        # Plot sample quads (first 100)
        n_sample = min(10000, len(quads))
        for i in range(n_sample):
            quad = quads[i]
            if len(quad) == 4:
                pts = np.array([[x[quad[j]], y[quad[j]]] for j in range(4)])
                pts = np.vstack([pts, pts[0]])  # Close the quad
                
                line = pg.PlotDataItem(
                    pts[:, 0], pts[:, 1],
                    pen=pg.mkPen('r', width=1, style=Qt.DashLine)
                )
                self.spatial_plot.addItem(line)
        
        self.spatial_plot.setTitle(f'ROIs and Sample Quads (showing {n_sample}/{len(quads)} quads)')
    
    def plot_area_distribution(self, areas):
        """Plot quad area distribution using utility"""
        self.area_figure.clear()
        ax = self.area_figure.add_subplot(111)
        
        plot_histogram_with_stats(
            ax, areas, bins=50,
            title='Distribution of Quad Areas',
            xlabel='Quad Area',
            ylabel='Frequency',
            show_mean=True,
            show_median=True
        )
        
        self.area_canvas.draw()
    
    def plot_aspect_ratios(self, x, y, quads):
        """Plot aspect ratio distribution using utility"""
        self.aspect_figure.clear()
        ax = self.aspect_figure.add_subplot(111)
        
        if len(quads) > 0:
            aspect_ratios = []
            for quad in quads:
                if len(quad) == 4:
                    pts = np.array([[x[quad[j]], y[quad[j]]] for j in range(4)])
                    d1 = np.linalg.norm(pts[0] - pts[2])
                    d2 = np.linalg.norm(pts[1] - pts[3])
                    aspect = max(d1, d2) / (min(d1, d2) + 1e-6)
                    aspect_ratios.append(aspect)
            
            if aspect_ratios:
                plot_histogram_with_stats(
                    ax, np.array(aspect_ratios), bins=50,
                    title='Distribution of Quad Aspect Ratios',
                    xlabel='Aspect Ratio',
                    ylabel='Frequency',
                    show_mean=True,
                    show_median=False
                )
        
        self.aspect_canvas.draw()
    
    def plot_spatial_participation(self, x, y, quad_idx):
        """Plot spatial distribution of neurons involved in quad generation"""
        self.participation_figure.clear()
        
        if len(quad_idx) == 0:
            ax = self.participation_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No quads generated', ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            self.participation_canvas.draw()
            return
        
        # Count how many times each neuron appears in quads
        neuron_counts = np.zeros(len(x), dtype=int)
        
        for quad in quad_idx:
            if len(quad) == 4:
                neuron_counts[quad] += 1
        
        # Create 2x1 subplot
        fig = self.participation_figure
        
        # Top: Heatmap
        ax1 = fig.add_subplot(2, 1, 1)
        self._plot_participation_heatmap(ax1, x, y, neuron_counts, 'Quad Participation per Neuron')
        
        # Bottom: Radial distribution
        ax2 = fig.add_subplot(2, 1, 2)
        self._plot_radial_distribution(ax2, x, y, neuron_counts, 'Center vs Periphery')
        
        fig.tight_layout()
        self.participation_canvas.draw()
    
    def _plot_participation_heatmap(self, ax, x, y, counts, title):
        """Plot 2D heatmap of neuron participation"""
        # Create 2D histogram weighted by counts
        bins = 30
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=counts)
        
        # Normalize by total neurons in each bin for participation rate
        H_total, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            participation_rate = H / H_total
            participation_rate = np.nan_to_num(participation_rate)
        
        im = ax.imshow(participation_rate.T, origin='lower', cmap='hot', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      aspect='auto', interpolation='bilinear')
        
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        ax.set_title(f'{title}\nAvg Quads per Neuron', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = self.participation_figure.colorbar(im, ax=ax)
        cbar.set_label('Avg Quads/Neuron', fontsize=9)
    
    def _plot_radial_distribution(self, ax, x, y, counts, title):
        """Plot participation vs distance from center"""
        # Calculate center
        cx = np.mean(x)
        cy = np.mean(y)
        
        # Calculate distances from center
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Bin by distance
        n_bins = 20
        max_dist = np.max(distances)
        bins = np.linspace(0, max_dist, n_bins + 1)
        
        bin_indices = np.digitize(distances, bins)
        
        # Calculate average participation in each bin
        avg_participation = []
        bin_centers = []
        
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_participation.append(np.mean(counts[mask]))
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        if bin_centers:
            ax.plot(bin_centers, avg_participation, 'o-', linewidth=2, markersize=6, color='steelblue')
            ax.fill_between(bin_centers, 0, avg_participation, alpha=0.3, color='steelblue')
            ax.set_xlabel('Distance from Center (pixels)', fontsize=10)
            ax.set_ylabel('Avg Quads per Neuron', fontsize=10)
            ax.set_title(f'{title}\nQuad Participation by Distance', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_dist)
            ax.set_ylim(0, None)
    
    def reset_params(self):
        """Reset parameters to defaults"""
        self.min_area_spin.setValue(10.0)
        self.height_percentile_spin.setValue(90.0)
        self.min_third_spin.setValue(3)
    
    def rerun_current_session(self):
        """Re-run quad generation for current session with new parameters"""
        if not self.current_session or not self.current_animal:
            return
        
        reply = QMessageBox.question(
            self, 'Confirm Re-run',
            f'Re-run quad generation for Animal {self.current_animal}, Session {self.current_session}?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config.min_tri_area = self.min_area_spin.value()
            self.config.height_percentile = self.height_percentile_spin.value()
            self.config.min_third_points_per_diagonal = self.min_third_spin.value()
            
            QMessageBox.information(
                self, 'Re-run',
                'Re-running quad generation...\n(Implementation needed)'
            )
    
    def rerun_all_sessions(self):
        """Re-run quad generation for all sessions with new parameters"""
        reply = QMessageBox.question(
            self, 'Confirm Re-run All',
            'Re-run quad generation for ALL sessions with current parameters?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config.min_tri_area = self.min_area_spin.value()
            self.config.height_percentile = self.height_percentile_spin.value()
            self.config.min_third_points_per_diagonal = self.min_third_spin.value()
            
            QMessageBox.information(
                self, 'Re-run All',
                'Re-running quad generation for all sessions...\n(Implementation needed)'
            )
    
    def export_summary(self):
        """Export summary statistics to text file"""
        QMessageBox.information(self, 'Export', 'Export functionality coming soon!')
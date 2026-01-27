"""
Step 2 Results Viewer: Matching Inspector (REFACTORED)
Visualize and analyze quad matching results between session pairs

CHANGES FROM ORIGINAL:
- Uses shared viewer utilities for UI creation
- Simplified file loading with error handling
- Reduced code duplication
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QGroupBox, QFormLayout, QDoubleSpinBox, 
                             QMessageBox, QSplitter, QTabWidget, QWidget)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from utilities import *

class Step2Viewer(QDialog):
    """Viewer for Step 2: Matching Results"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_animal = None
        self.current_pair = None
        self.match_data = {}
        self.animal_pairs = {}
        
        self.setWindowTitle('Step 2: Quad Matching Results')
        self.resize(1400, 900)
        
        self.init_ui()
        self.load_results()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Top controls using utility
        self.controls = create_top_controls(
            'Session Pair Selection',
            combos=[
                ('Animal', self.on_animal_changed),
                ('Pair', self.on_pair_changed),
            ],
            refresh_callback=self.load_results
        )
        self.animal_combo, self.pair_combo = self.controls.combos
        layout.addWidget(self.controls)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Stats and match table
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right: Visualizations
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Bottom buttons using utility
        buttons = create_action_buttons({
            '💾 Export Summary': self.export_summary,
            '✖ Close': self.accept
        })
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def create_left_panel(self):
        """Create left panel with stats and match table"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Match statistics using utility - UPDATED
        stat_labels = [
            ('n_raw_matches', 'Raw Matches'),           # NEW
            ('n_filtered_matches', 'Filtered Matches'), # NEW
            ('filter_rate', 'Filter Pass Rate'),        # NEW
            ('threshold_used', 'Threshold Used'),       # NEW
            ('avg_distance', 'Avg Descriptor Distance'),
            ('median_distance', 'Median Distance'),
            ('position_error', 'Avg Position Error'),
        ]
        self.stats_group, self.stat_labels = create_stat_group(
            'Match Statistics', stat_labels
        )
        layout.addWidget(self.stats_group)
        
        # Match table
        table_group = QGroupBox('Top Matches')
        table_layout = QVBoxLayout()
        
        self.match_table = QTableWidget()
        self.match_table.setColumnCount(6)
        self.match_table.setHorizontalHeaderLabels([
            'Ref Quad', 'Target Quad', 'Distance', 'Shape Diff', 'Δx', 'Δy'
        ])
        table_layout.addWidget(self.match_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        return widget
        
    def create_right_panel(self):
        """Create right panel with visualizations"""
        tabs = QTabWidget()
        
        # Tab 1: Spatial matches using utility
        spatial_tab, self.spatial_plot = create_pyqtgraph_tab('Spatial Matches')
        setup_pyqtgraph_plot(
            self.spatial_plot,
            'Matched ROIs',
            'X Coordinate',
            'Y Coordinate'
        )
        self.spatial_plot.setAspectLocked(True)
        tabs.addTab(spatial_tab, '📍 Spatial Matches')
        
        # Tab 2: Score distribution using utility
        score_tab, self.score_figure, self.score_canvas = create_matplotlib_tab(
            'Score Distribution'
        )
        tabs.addTab(score_tab, '📊 Score Distribution')
        
        # Tab 3: Displacement analysis using utility
        disp_tab, self.disp_figure, self.disp_canvas = create_matplotlib_tab(
            'Displacement Analysis'
        )
        tabs.addTab(disp_tab, '📐 Displacement')
        
        # Tab 4: Spatial participation using utility
        participation_tab, self.participation_figure, self.participation_canvas = create_matplotlib_tab(
            'Spatial Participation'
        )
        tabs.addTab(participation_tab, '🎯 Spatial Participation')
        
        # Tab 5: Parameters
        config_tab = self.create_config_tab()
        tabs.addTab(config_tab, '⚙️ Parameters')
        
        # Tab 6: Statistics using utility
        stats_tab, self.stats_text, self.load_pipeline_stats = create_stats_tab(2, self.config.output_dir)

        tabs.addTab(stats_tab, '📋 Statistics')
        
        return tabs
        
    def create_config_tab(self):
        """Create configuration tab for parameter adjustment"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info = QLabel('Adjust threshold and re-run matching for current pair')
        info.setStyleSheet('font-weight: bold; padding: 10px;')
        layout.addWidget(info)
        
        # Parameter form
        form = QFormLayout()
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 1.0)
        self.threshold_spin.setValue(self.config.threshold if self.config.threshold else 0.01)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.001)
        form.addRow('Descriptor Threshold:', self.threshold_spin)
        
        layout.addLayout(form)
        
        # UPDATED info text
        info_text = QLabel(
            'The threshold is the maximum descriptor distance for a match.\n'
            'This value comes from Step 1.5 calibration.\n\n'
            'Lower threshold = stricter matching, fewer matches.\n'
            'Higher threshold = looser matching, more matches.\n\n'
            'Matches also pass through a consistency filter\n'
            'that checks geometric scale consistency.'
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet('color: gray; font-size: 10px; padding: 10px;')
        layout.addWidget(info_text)
        
        # Action buttons
        buttons = create_action_buttons({
            '↺ Reset to Step 1.5 Value': self.reset_params,
            '▶️ Re-run Current Pair': self.rerun_current_pair,
            '▶️ Re-run All Pairs': self.rerun_all_pairs,
        }, add_stretch=False)
        layout.addWidget(buttons)
        
        layout.addStretch()
        
        return widget
     
    def load_results(self):
        """Load Step 2 matching results"""
        step2_dir = get_step_results_dir(self.config.output_dir, 2)
        
        print(f"\n{'='*60}")
        print(f"Loading Step 2 results from: {step2_dir}")
        
        if not step2_dir.exists():
            show_no_results_error(self, step2_dir, 'Step 2')
            return
            
        # Find match files using utility
        match_files = scan_results_directory(
            step2_dir,
            '*_matches_light.npz',
            verbose=True
        )
        
        if not match_files:
            show_no_files_error(self, step2_dir, '*_matches_light.npz', 'Step 2')
            return
            
        print(f"Found {len(match_files)} match files")
            
        # Clear previous data
        self.match_data.clear()
        self.animal_pairs.clear()
        self.animal_combo.clear()
        self.pair_combo.clear()
        
        # Load each match file
        loaded_count = 0
        for match_file in match_files:
            # Load using utility
            data = load_npz_safely(match_file, verbose=True)
            if data is None:
                continue
            
            # Extract animal and session pair from filename
            # Format: {animal}_{sess1}_to_{sess2}_matches_light.npz
            stem = match_file.stem.replace('_matches_light', '')
            parts = stem.split('_to_')
            
            if len(parts) == 2:
                left_parts = parts[0].split('_')
                if len(left_parts) >= 2:
                    animal_id = left_parts[0]
                    sess1 = '_'.join(left_parts[1:])
                    sess2 = parts[1]
                    
                    if animal_id not in self.animal_pairs:
                        self.animal_pairs[animal_id] = []
                    
                    pair_key = f"{sess1}→{sess2}"
                    self.animal_pairs[animal_id].append(pair_key)
                    
                    full_key = f"{animal_id}_{pair_key}"
                    self.match_data[full_key] = {
                        'file': match_file,
                        'data': data,
                        'animal': animal_id,
                        'sess1': sess1,
                        'sess2': sess2
                    }
                    
                    loaded_count += 1
                    print(f"  ✓ Loaded as {full_key}")
                
        print(f"{'='*60}")
        print(f"Successfully loaded {loaded_count} session pairs")
        print(f"{'='*60}\n")
        
        if loaded_count == 0:
            QMessageBox.warning(
                self, 'Load Failed',
                f'Could not load any match files from:\n{step2_dir}'
            )
            return
                
        # Populate combos
        animals = sorted(self.animal_pairs.keys())
        self.animal_combo.addItems(animals)
        
        if animals:
            self.animal_combo.setCurrentIndex(0)
        
        # Load pipeline statistics
        self.load_pipeline_stats()

    def on_animal_changed(self, animal_id):
        """Handle animal selection change"""
        if not animal_id or animal_id not in self.animal_pairs:
            return
            
        self.current_animal = animal_id
        
        self.pair_combo.clear()
        pairs = sorted(self.animal_pairs[animal_id])
        self.pair_combo.addItems(pairs)
        
        if pairs:
            self.pair_combo.setCurrentIndex(0)
            
    def on_pair_changed(self, pair_key):
        """Handle pair selection change"""
        if not pair_key or not self.current_animal:
            return
            
        self.current_pair = pair_key
        full_key = f"{self.current_animal}_{pair_key}"
        
        if full_key in self.match_data:
            self.display_pair(full_key)
            
    def display_pair(self, full_key):
        """Display match data for selected pair"""
        match_info = self.match_data[full_key]
        data = match_info['data']
        
        print(f"\nDisplaying pair: {full_key}")
        
        # Extract match indices (N, 8) - first 4 are ref quad, last 4 are target quad
        if 'match_indices' not in data:
            QMessageBox.warning(
                self, 'No Matches',
                f'Could not find match_indices in {full_key}\n\n'
                f'Available fields: {", ".join(data.keys())}'
            )
            return
        
        match_indices = data['match_indices']
        n_filtered = len(match_indices)
        
        # Get raw/filtered counts from new fields
        n_raw = int(data.get('n_raw_matches', n_filtered))
        threshold_used = float(data.get('threshold_used', 0))
        
        # Extract centroids
        ref_centroids = data['ref_centroids']
        target_centroids = data['target_centroids']
        
        # Use distances directly if available (NEW)
        if 'distances' in data and len(data['distances']) > 0:
            distances = data['distances']
        elif 'ref_descriptors' in data and 'tgt_descriptors' in data:
            ref_descriptors = data['ref_descriptors']
            tgt_descriptors = data['tgt_descriptors']
            if len(ref_descriptors) > 0 and len(tgt_descriptors) > 0:
                distances = np.linalg.norm(ref_descriptors - tgt_descriptors, axis=1)
            else:
                distances = np.zeros(n_filtered)
        else:
            distances = np.zeros(n_filtered)
        
        # Extract coordinates (centroids are (y, x))
        sess1_y = ref_centroids[:, 0]
        sess1_x = ref_centroids[:, 1]
        sess2_y = target_centroids[:, 0]
        sess2_x = target_centroids[:, 1]
        
        print(f"  Raw matches: {n_raw}")
        print(f"  Filtered matches: {n_filtered}")
        print(f"  Threshold: {threshold_used:.4f}")
        print(f"  Ref neurons: {len(ref_centroids)}")
        print(f"  Target neurons: {len(target_centroids)}")
        
        # Calculate position errors
        pos_errors = []
        for i in range(min(1000, n_filtered)):
            match = match_indices[i]
            ref_quad = match[:4]
            tgt_quad = match[4:]
            
            ref_cx = np.mean(sess1_x[ref_quad])
            ref_cy = np.mean(sess1_y[ref_quad])
            tgt_cx = np.mean(sess2_x[tgt_quad])
            tgt_cy = np.mean(sess2_y[tgt_quad])
            
            dx = tgt_cx - ref_cx
            dy = tgt_cy - ref_cy
            pos_errors.append(np.sqrt(dx**2 + dy**2))
        
        # Update statistics - UPDATED
        filter_rate = n_filtered / n_raw * 100 if n_raw > 0 else 0
        
        stats = {
            'n_raw_matches': f"{n_raw:,}",
            'n_filtered_matches': f"{n_filtered:,}",
            'filter_rate': f"{filter_rate:.1f}%",
            'threshold_used': f"{threshold_used:.4f}",
        }
        
        if len(distances) > 0:
            stats['avg_distance'] = f"{np.mean(distances):.4f}"
            stats['median_distance'] = f"{np.median(distances):.4f}"
        
        if pos_errors:
            stats['position_error'] = f"{np.mean(pos_errors):.2f} px"
        
        update_stat_labels(self.stat_labels, stats)
        
        # Update match table
        self._update_match_table(match_indices, distances, 
                                sess1_x, sess1_y, sess2_x, sess2_y)
        
        # Update plots
        self.plot_spatial_matches(sess1_x, sess1_y, sess2_x, sess2_y, match_indices)
        self.plot_score_distribution(distances)
        self.plot_displacement(sess1_x, sess1_y, sess2_x, sess2_y, match_indices)
        self.plot_spatial_participation(sess1_x, sess1_y, sess2_x, sess2_y, match_indices)
        
        print(f"Display complete\n")

    def _update_match_table(self, match_indices, distances, x1, y1, x2, y2):
        """Update the match details table"""
        n_matches = len(match_indices)
        n_display = min(100, n_matches)
        
        self.match_table.setRowCount(n_display)
        
        for i in range(n_display):
            match = match_indices[i]
            distance = distances[i] if i < len(distances) else 0
            
            ref_quad = match[:4]
            tgt_quad = match[4:]
            
            # Calculate displacement
            ref_cx = np.mean(x1[ref_quad])
            ref_cy = np.mean(y1[ref_quad])
            tgt_cx = np.mean(x2[tgt_quad])
            tgt_cy = np.mean(y2[tgt_quad])
            
            dx = tgt_cx - ref_cx
            dy = tgt_cy - ref_cy
            
            ref_str = f"[{','.join(map(str, ref_quad))}]"
            tgt_str = f"[{','.join(map(str, tgt_quad))}]"
            
            self.match_table.setItem(i, 0, QTableWidgetItem(ref_str))
            self.match_table.setItem(i, 1, QTableWidgetItem(tgt_str))
            self.match_table.setItem(i, 2, QTableWidgetItem(f"{distance:.4f}"))
            self.match_table.setItem(i, 3, QTableWidgetItem(f"{distance:.4f}"))  # Same as distance now
            self.match_table.setItem(i, 4, QTableWidgetItem(f"{dx:.2f}"))
            self.match_table.setItem(i, 5, QTableWidgetItem(f"{dy:.2f}"))
            
    def plot_spatial_matches(self, x1, y1, x2, y2, match_indices):
        """Plot spatial distribution of matched ROIs"""
        self.spatial_plot.clear()
        
        if len(x1) == 0 or len(x2) == 0:
            return
        
        # Plot session 1 ROIs (blue)
        scatter1 = pg.ScatterPlotItem(
            x1, y1,
            size=6,
            brush=pg.mkBrush(0, 100, 255, 100),
            pen=pg.mkPen('w', width=0.5),
            symbol='o'
        )
        self.spatial_plot.addItem(scatter1)
        
        # Plot session 2 ROIs (red)
        scatter2 = pg.ScatterPlotItem(
            x2, y2,
            size=6,
            brush=pg.mkBrush(255, 100, 0, 100),
            pen=pg.mkPen('w', width=0.5),
            symbol='s'
        )
        self.spatial_plot.addItem(scatter2)
        
        # Draw match lines between quad centroids (sample)
        n_sample = min(10000, len(match_indices))
        for match in match_indices[:n_sample]:
            ref_quad = match[:4]
            tgt_quad = match[4:]
            
            ref_cx = np.mean(x1[ref_quad])
            ref_cy = np.mean(y1[ref_quad])
            tgt_cx = np.mean(x2[tgt_quad])
            tgt_cy = np.mean(y2[tgt_quad])
            
            line = pg.PlotDataItem(
                [ref_cx, tgt_cx],
                [ref_cy, tgt_cy],
                pen=pg.mkPen('g', width=1, style=Qt.DashLine)
            )
            self.spatial_plot.addItem(line)
        
        self.spatial_plot.setTitle(
            f'Ref (○ blue) → Target (□ red) | '
            f'Showing {n_sample}/{len(match_indices)} quad matches'
        )

    def plot_score_distribution(self, distances):
        """Plot descriptor distance distribution"""
        self.score_figure.clear()
        ax = self.score_figure.add_subplot(111)
        
        if len(distances) > 0:
            plot_histogram_with_stats(
                ax, distances, bins=50,
                title='Distribution of Descriptor Distances',
                xlabel='Descriptor Distance',
                ylabel='Frequency',
                show_mean=True,
                show_median=True
            )
                
        self.score_canvas.draw()
        
    def plot_displacement(self, x1, y1, x2, y2, match_indices):
        """Plot displacement vectors"""
        self.disp_figure.clear()
        ax = self.disp_figure.add_subplot(111)
        
        if len(match_indices) > 0 and len(x1) > 0 and len(x2) > 0:
            dx_list = []
            dy_list = []
            
            for match in match_indices[:500]:
                ref_quad = match[:4]
                tgt_quad = match[4:]
                
                ref_cx = np.mean(x1[ref_quad])
                ref_cy = np.mean(y1[ref_quad])
                tgt_cx = np.mean(x2[tgt_quad])
                tgt_cy = np.mean(y2[tgt_quad])
                
                dx = tgt_cx - ref_cx
                dy = tgt_cy - ref_cy
                dx_list.append(dx)
                dy_list.append(dy)
            
            if dx_list:
                ax.scatter(dx_list, dy_list, alpha=0.5, s=20, c='steelblue')
                ax.set_xlabel('Δx (pixels)', fontsize=12)
                ax.set_ylabel('Δy (pixels)', fontsize=12)
                ax.set_title('Displacement Vectors (Ref → Target Quad Centroids)',
                            fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
                
                mean_dx = np.mean(dx_list)
                mean_dy = np.mean(dy_list)
                ax.plot(mean_dx, mean_dy, 'r*', markersize=15,
                       label=f'Mean: ({mean_dx:.2f}, {mean_dy:.2f})')
                ax.legend()
        
        self.disp_canvas.draw()
    
    def plot_spatial_participation(self, x1, y1, x2, y2, match_indices):
        """Plot spatial distribution of neurons involved in quad matches"""
        self.participation_figure.clear()
        
        if len(match_indices) == 0 or len(x1) == 0 or len(x2) == 0:
            return
        
        # Count how many times each neuron appears in matches
        ref_counts = np.zeros(len(x1), dtype=int)
        tgt_counts = np.zeros(len(x2), dtype=int)
        
        for match in match_indices:
            ref_quad = match[:4]
            tgt_quad = match[4:]
            ref_counts[ref_quad] += 1
            tgt_counts[tgt_quad] += 1
        
        # Create 2x2 subplot
        fig = self.participation_figure
        
        # Top left: Reference session heatmap
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_participation_heatmap(ax1, x1, y1, ref_counts, 'Reference Session')
        
        # Top right: Target session heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_participation_heatmap(ax2, x2, y2, tgt_counts, 'Target Session')
        
        # Bottom left: Reference radial distribution
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_radial_distribution(ax3, x1, y1, ref_counts, 'Reference Session')
        
        # Bottom right: Target radial distribution
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_radial_distribution(ax4, x2, y2, tgt_counts, 'Target Session')
        
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
        ax.set_title(f'{title}\nAvg Match Participation per Neuron', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = self.participation_figure.colorbar(im, ax=ax)
        cbar.set_label('Avg Matches/Neuron', fontsize=9)
    
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
            ax.set_ylabel('Avg Matches per Neuron', fontsize=10)
            ax.set_title(f'{title}\nCenter vs Periphery Participation', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_dist)
            ax.set_ylim(0, None)
        
    def reset_params(self):
        """Reset parameters to defaults"""
        self.threshold_spin.setValue(0.15)
        
    def rerun_current_pair(self):
        """Re-run matching for current pair"""
        if not self.current_pair or not self.current_animal:
            return
            
        reply = QMessageBox.question(
            self, 'Confirm Re-run',
            f'Re-run matching for {self.current_animal}: {self.current_pair}?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config.threshold = self.threshold_spin.value()
            QMessageBox.information(
                self, 'Re-run',
                'Re-running matching...\n(Implementation needed)'
            )
            
    def rerun_all_pairs(self):
        """Re-run matching for all pairs"""
        reply = QMessageBox.question(
            self, 'Confirm Re-run All',
            'Re-run matching for ALL pairs?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config.threshold = self.threshold_spin.value()
            QMessageBox.information(
                self, 'Re-run All',
                'Re-running all matching...\n(Implementation needed)'
            )
            
    def export_summary(self):
        """Export summary statistics"""
        QMessageBox.information(self, 'Export', 'Export functionality coming soon!')
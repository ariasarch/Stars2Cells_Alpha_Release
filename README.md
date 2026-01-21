# Stars2Cells: Neuron Tracking Across Recording Sessions

> **Note**: This is an early release version of the Stars2Cells pipeline. Features and documentation are actively being developed and refined.

## Overview

Stars2Cells is a robust neuron tracking pipeline that enables researchers to identify and track individual neurons across multiple calcium imaging sessions. Inspired by astronomical star-matching techniques, this method uses geometric pattern recognition to reliably match neurons even when there are shifts, rotations, or missing cells between recordings.

## When to Use This Pipeline

- **Longitudinal studies**: Track the same neurons across days, weeks, or months
- **Pre/post intervention experiments**: Match neurons before and after experimental manipulations
- **Multi-session recordings**: Maintain neuron identity across separate imaging sessions
- **FOV registration**: Handle field-of-view shifts between recording sessions

## Pipeline Overview

The complete workflow consists of five sequential steps:

1. **Quad Generation** - Create geometric patterns from neuron positions
2. **Threshold Generation** - Determine optimal matching criteria
3. **Matching Generator** - Find candidate neuron matches between sessions
4. **RANSAC Refinement** - Robustly estimate transformation parameters
5. **Neuron Matching** - Finalize neuron identities across sessions

---

## Step 1: Quad Generation

### Purpose
Generate geometric patterns (quadrilaterals) from neuron centroids to create a position-invariant representation of cellular arrangements.

### What It Does
- Takes neuron centroid coordinates from each recording session
- Identifies sets of 4 nearby neurons (quads)
- Computes geometric invariants (ratios of distances/areas) for each quad
- Creates a library of distinctive spatial patterns

### Input Requirements
- `.npy` files containing neuron centroid data (see Input Data Format section)
- Minimum of 4 neurons per session
- All sessions should follow the `{animal_id}_{session_id}.npy` naming convention

### Output
- Quad library for Session A
- Quad library for Session B
- Geometric hash values for fast matching

### Key Parameters
- **max_quad_distance**: Maximum distance between neurons to form a quad (default: ~50 pixels)
- **min_neurons**: Minimum neurons required per quad (typically 4)

### Usage Notes
- Higher neuron density = more quads generated
- Trade-off between computational cost and matching robustness
- Recommended to use neurons with high spatial information (avoid edge artifacts)

---

## Step 2: Threshold Generation

### Purpose
Determine the optimal similarity threshold for distinguishing true matches from false positives.

### What It Does
- Compares geometric invariants between Session A and Session B quads
- Generates a distribution of similarity scores
- Calculates statistical thresholds based on the score distribution
- Identifies the decision boundary for accepting matches

### Input Requirements
- Quad libraries from Step 1

### Output
- Similarity threshold value
- Score distribution statistics
- Recommended confidence intervals

### Key Parameters
- **percentile**: Confidence level for threshold (default: 95th or 99th percentile)
- **method**: Statistical method ('percentile', 'mad', or 'iqr')

### Usage Notes
- More stringent thresholds (higher percentile) = fewer false matches but may miss some true matches
- Less stringent thresholds = more matches but increased false positive rate
- Visualize score distributions to verify appropriate threshold selection

---

## Step 3: Matching Generator

### Purpose
Use quad-based geometric matching to find candidate neuron correspondences between sessions.

### What It Does
- Compares all quads from Session A against Session B
- Identifies quads with similar geometric properties (below threshold)
- Generates voting matrix: each matched quad "votes" for specific neuron pairs
- Creates initial neuron correspondence list

### Input Requirements
- Quad libraries from Step 1
- Similarity threshold from Step 2

### Output
- Voting matrix (neurons × neurons)
- Initial match candidates with confidence scores
- Consensus matches (neurons with high vote counts)

### Key Parameters
- **min_votes**: Minimum votes required to consider a match valid (default: 3-5)
- **vote_threshold**: Minimum voting percentage for acceptance

### Usage Notes
- Higher vote counts indicate more reliable matches
- Neurons near FOV edges may have fewer votes (fewer quads)
- Review voting distribution to assess overall matching quality

---

## Step 4: RANSAC Refinement

### Purpose
Robustly estimate the global geometric transformation between sessions while eliminating outliers.

### What It Does
- Uses RANSAC (Random Sample Consensus) algorithm
- Estimates affine transformation parameters (translation, rotation, scaling, shear)
- Identifies and removes outlier matches that don't fit the global transformation
- Refines neuron correspondences based on transformation consistency

### Input Requirements
- Initial match candidates from Step 3
- Neuron coordinates from both sessions

### Output
- Transformation matrix (2×3 affine transform)
- Inlier matches (neurons consistent with transformation)
- Outlier matches (likely false positives)
- Transformation residuals and quality metrics

### Key Parameters
- **max_iterations**: RANSAC iterations (default: 1000-5000)
- **inlier_threshold**: Maximum allowable spatial error for inliers (default: 2-5 pixels)
- **min_samples**: Minimum matches needed for transformation estimation (default: 4)

### Usage Notes
- Critical for handling FOV shifts and rotations
- Removes systematic matching errors
- Check residual distribution to verify good transformation fit
- Visualize before/after transformation to confirm alignment

---

## Step 5: Neuron Matching

### Purpose
Finalize neuron identities across sessions and create the tracking results.

### What It Does
- Takes inlier matches from RANSAC
- Resolves any one-to-many or many-to-one matches
- Assigns unique neuron IDs that persist across sessions
- Generates comprehensive matching report

### Input Requirements
- RANSAC inliers from Step 4
- Original neuron data from both sessions

### Output
- **Match table**: CSV with columns [SessionA_ID, SessionB_ID, Confidence, Spatial_Error]
- **Unmatched neurons**: Lists of neurons unique to each session
- **Matching statistics**: Success rate, accuracy metrics
- **Visualization files** (optional): Overlay plots showing matched neurons

### Key Parameters
- **conflict_resolution**: Method for handling ambiguous matches ('highest_confidence', 'nearest_neighbor')
- **min_confidence**: Minimum confidence score to accept a match

### Usage Notes
- Some neurons may remain unmatched (cell loss, new cells, FOV differences)
- Track matching success rate across the pipeline
- Manually review edge cases or low-confidence matches if needed

---

## Complete Workflow Example

```python
# Prepare your input directory with .npy files
# input_dir/
#   ├── 408021_758519303.npy  (Session 1)
#   ├── 408021_758519304.npy  (Session 2)
#   └── 408021_758519305.npy  (Session 3)

# Step 1: Generate quads for all session pairs
python step_1_quad_generation.py \
    --input-dir input_dir/ \
    --output quads_output/

# Step 1.5: Calculate optimal threshold
python step_1_5_threshold_generation.py \
    --quads quads_output/ \
    --output threshold.txt

# Step 2: Generate initial matches
python step_2_matching_generator.py \
    --quads quads_output/ \
    --threshold threshold.txt \
    --output matches_initial/

# Step 2.5: RANSAC refinement
python step_2_5_RANSAC.py \
    --matches matches_initial/ \
    --input-dir input_dir/ \
    --output matches_refined/

# Step 3: Finalize matches
python step_3_neuron_matching.py \
    --matches matches_refined/ \
    --output final_matches/
```

---

## Quality Control Checkpoints

### After Step 1 (Quad Generation)
- ✓ Verify reasonable number of quads generated (typically 10-100× number of neurons)
- ✓ Check quad spatial distribution across FOV

### After Step 2 (Threshold Generation)
- ✓ Examine score distribution histogram
- ✓ Ensure clear separation between matches and non-matches
- ✓ Verify threshold falls in appropriate range

### After Step 3 (Matching Generator)
- ✓ Review voting matrix for clear diagonal pattern
- ✓ Check that most neurons have reasonable vote counts
- ✓ Identify neurons with suspiciously low votes

### After Step 4 (RANSAC)
- ✓ Verify transformation parameters are reasonable
- ✓ Check inlier percentage (typically >70% for good recordings)
- ✓ Examine residual distribution (should be tight, <2-5 pixels)
- ✓ Visualize transformed overlay

### After Step 5 (Final Matching)
- ✓ Calculate matching success rate
- ✓ Check for unexpected patterns in unmatched neurons
- ✓ Verify spatial distribution of matches across FOV

---

## Expected Performance

**Typical Results:**
- **Matching success rate**: 70-95% (depends on FOV stability)
- **False positive rate**: <5% (with proper thresholding)
- **Spatial accuracy**: <2 pixels RMS error
- **Processing time**: 1-5 minutes per session pair (1000 neurons)

**Factors Affecting Performance:**
- FOV stability between sessions
- Neuron density
- Imaging quality and SNR
- Amount of FOV shift/rotation
- Cell turnover rate

---

## Troubleshooting

### Low Matching Rate (<50%)

**Possible causes:**
- Excessive FOV shift between sessions
- High cell turnover (many new/lost neurons)
- Poor neuron detection quality
- Threshold too stringent

**Solutions:**
- Verify neuron detection quality in both sessions
- Relax similarity threshold (Step 2)
- Increase RANSAC iterations and inlier threshold
- Check for systematic FOV rotations

### High False Positive Rate

**Possible causes:**
- Threshold too permissive
- Insufficient quad diversity
- Low neuron count

**Solutions:**
- Increase threshold stringency
- Adjust max_quad_distance parameter
- Increase min_votes requirement
- Manually review match confidence scores

### Processing Time Too Long

**Possible causes:**
- Too many quads generated
- Excessive RANSAC iterations

**Solutions:**
- Reduce max_quad_distance
- Pre-filter low-quality neurons
- Reduce RANSAC max_iterations
- Consider downsampling for initial pass

---

## Input Data Format

### File Requirements

The Stars2Cells pipeline requires neuron centroid data in `.npy` format. Each recording session must be saved as a separate file following the standardized structure below.

### File Naming Convention

**Format**: `{animal_id}_{session_id}.npy`

**Requirements**:
- Both `animal_id` and `session_id` must be numeric strings
- Use underscore (`_`) as separator
- File extension must be `.npy`

**Examples**:
- ✓ `408021_758519303.npy`
- ✓ `12345_67890.npy`
- ✗ `mouse1_session1.npy` (non-numeric IDs)
- ✗ `408021-758519303.npy` (wrong separator)

### File Structure

Each `.npy` file must contain a Python dictionary saved with NumPy's `save()` function using `allow_pickle=True`.

**Required Dictionary Keys** (5 total):

1. **`centroids_x`** (array/list)
   - X-coordinates of neuron centroids in pixels
   - Type: numpy array or list of numbers
   - Length: N (number of neurons)

2. **`centroids_y`** (array/list)
   - Y-coordinates of neuron centroids in pixels
   - Type: numpy array or list of numbers
   - Length: N (must match `centroids_x`)

3. **`roi_ids`** (array/list)
   - Unique identifier for each neuron/ROI
   - Type: numpy array or list (integers or strings)
   - Length: N (must match centroids)
   - Each ID should be unique within the session

4. **`subject_id`** (string or int)
   - Subject/animal identifier
   - Should match the `animal_id` from the filename

5. **`session_id`** (string or int)
   - Session identifier
   - Should match the `session_id` from the filename

### Creating Input Files

```python
import numpy as np

# Your neuron data
centroids_x = [123.4, 456.7, 789.0, 234.1]  # X coordinates in pixels
centroids_y = [234.5, 567.8, 890.1, 345.2]  # Y coordinates in pixels
roi_ids = [0, 1, 2, 3]                       # Unique ID for each neuron

# Create the required dictionary
data = {
    'centroids_x': np.array(centroids_x),
    'centroids_y': np.array(centroids_y),
    'roi_ids': np.array(roi_ids),
    'subject_id': '408021',      # Animal ID (matches filename)
    'session_id': '758519303'    # Session ID (matches filename)
}

# Save to file with proper naming
filename = f"{data['subject_id']}_{data['session_id']}.npy"
np.save(filename, data, allow_pickle=True)
```

### Loading and Verifying Files

```python
import numpy as np

# Load the file
data = np.load('408021_758519303.npy', allow_pickle=True).item()

# Verify structure
required_keys = ['centroids_x', 'centroids_y', 'roi_ids', 'subject_id', 'session_id']
assert all(key in data for key in required_keys), "Missing required keys"

# Verify array lengths match
n_neurons = len(data['centroids_x'])
assert len(data['centroids_y']) == n_neurons, "centroids_y length mismatch"
assert len(data['roi_ids']) == n_neurons, "roi_ids length mismatch"
assert n_neurons >= 4, f"Need at least 4 neurons, found {n_neurons}"

print(f"✓ File valid: {n_neurons} neurons")
```

### Example File Contents

```python
{
    'centroids_x': array([123.4, 456.7, 789.0, 234.1]),
    'centroids_y': array([234.5, 567.8, 890.1, 345.2]),
    'roi_ids': array([0, 1, 2, 3]),
    'subject_id': '408021',
    'session_id': '758519303',
    
    # Optional: Additional fields allowed (will be ignored by pipeline)
    'footprints': ...,
    'traces': ...,
    'timestamps': ...,
}
```

### Data Requirements

- ✓ Array lengths: `len(centroids_x) == len(centroids_y) == len(roi_ids)`
- ✓ Minimum neurons: At least 4 neurons per session (sessions with <4 are skipped)
- ✓ Coordinate system: Pixel coordinates (typically 0,0 at top-left)
- ✓ Centroid precision: Float or integer values accepted
- ✓ ROI IDs: Must be unique within each session

### Directory Organization

Place all `.npy` files in a single input directory:

```
input_directory/
├── 408021_758519303.npy
├── 408021_758519304.npy
├── 408021_758519305.npy
└── 412345_758519400.npy
```

### Converting from Other Formats

**From Suite2P:**
- Extract `stat.mat` or ROI coordinates from output
- Convert to Python dict with required keys
- Save with `np.save(file, data, allow_pickle=True)`

**From CaImAn:**
- Use `cnm.estimates.coordinates` to get centroids
- Package into required dict format
- Save as `.npy` file

**From EXTRACT (MATLAB):**
- Parse `output.mat` for centroids
- Convert to numpy arrays in Python
- Create dict and save

### Validation Checklist

Before running the pipeline, verify each file:

- [ ] Filename matches pattern: `{animal_id}_{session_id}.npy`
- [ ] File loads successfully: `np.load(file, allow_pickle=True).item()`
- [ ] Returns a dictionary (not array or other type)
- [ ] Contains all 5 required keys
- [ ] `centroids_x`, `centroids_y`, `roi_ids` all have same length
- [ ] At least 4 neurons present
- [ ] `subject_id` and `session_id` in dict match filename

### Common Format Errors

**"Not enough neurons (X < 4)"**
- Session has fewer than 4 neurons and will be skipped

**"Missing required fields"**
- Dictionary doesn't contain all 5 required keys

**"NO MATCH: filename.npy"**
- Filename doesn't match `{animal_id}_{session_id}.npy` pattern

**"File is not a dictionary"**
- `.npy` file doesn't contain a dict (might be raw array)

### Important Notes

- The pipeline internally converts centroids to (y, x) order for processing
- Additional metadata fields are allowed but will be ignored
- ROI IDs don't need to be sequential (can be any unique values)
- Coordinate units are pixels (no conversion applied)
- All sessions for one animal should use a consistent coordinate system

---

## Citation and References

If you use Stars2Cells in your research, please cite:

*Manuscript in preparation*

**Methodological inspiration:**
- Teague, M.R. (1980). "Image analysis via the general theory of moments." *Journal of the Optical Society of America*
- Astronomical star pattern matching algorithms
- RANSAC: Fischler & Bolles (1981). "Random sample consensus." *Communications of the ACM*

---

## Support and Contributions

This is an early release version. For questions, bug reports, or feature requests, please contact the Neumaier Lab at the University of Washington.

---

## Acknowledgments

Developed by Ari-Peden Asarch at the Neumaier Lab at the University of Washington 

This pipeline was developed to address computational challenges in longitudinal calcium imaging studies, making robust neuron tracking accessible to the neuroscience community.

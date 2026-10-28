# HealthScan: Camera-Based Health Screening Proposal

## 1. Executive summary

HealthScan is a local, real-time, non-contact health-screening prototype. It
uses a webcam or phone-camera stream, MediaPipe Face Mesh landmarks, OpenCV
image processing, NumPy numerical operations, and SciPy signal processing to
estimate:

- heart rate from facial remote photoplethysmography (rPPG);
- fatigue from eye openness and blink behavior;
- stress-related indicators from blink behavior and eyebrow position;
- jaundice-related color changes in the sclera;
- anemia-related color changes in the lips.

The current implementation is a screening aid for experimentation and
education. It is not a clinical diagnostic system. In particular, the
repository does **not** contain a project-specific supervised learning dataset,
trained classifier, model-training script, learned weights, or clinical
validation results. The only learned component is MediaPipe's pretrained face
landmark model, which is used as a general-purpose computer-vision dependency.
The health indicators are calculated using signal-processing equations,
calibration, temporal averaging, and fixed thresholds.

The proposed next phase is to preserve the current local, privacy-preserving
pipeline while adding data collection, clinical labels, subject-independent
evaluation, confidence estimation, and—only after sufficient validated data—
supervised models for the screening tasks.

## 2. Problem statement and motivation

Many health measurements require dedicated equipment or a clinical visit.
Camera-based measurement may provide an inexpensive preliminary screening
interface when a user has access to a camera but not to specialized sensors.
However, facial video is strongly affected by illumination, camera exposure,
motion, camera quality, skin appearance, facial expression, and demographic
variation. Therefore, the responsible objective is not to diagnose disease
from a camera image. The objective is to extract stable physiological and
appearance-related signals, report them with uncertainty, and direct the user
to a qualified professional when an indicator is abnormal.

## 3. Current implementation and progress

### Completed functionality

1. **Camera acquisition and display**
   - `main.py` opens camera device `0` on Linux and a configurable MJPEG URL
     on other platforms.
   - Frames are resized to 640 x 480 and displayed in an OpenCV window.
   - `Q` exits and `R` resets the session.

2. **Face localization and alignment**
   - MediaPipe Face Mesh is configured for one face, refined landmarks,
     detection confidence 0.5, and tracking confidence 0.5.
   - `FaceGuide` uses the nose, ear landmarks, forehead, and chin to determine
     whether the face is centered and within a target size.
   - Measurement starts only after 30 consecutive aligned frames.

3. **Personal calibration**
   - `Calibrator` collects 90 aligned frames, approximately three seconds at
     30 frames/second.
   - Median cheek HSV values, green-channel value, baseline eye-aspect ratio,
     and baseline eyebrow distance are stored for the current person/session.

4. **Heart-rate estimation**
   - A valid facial skin ROI is selected from forehead, cheek, or chin zones.
   - Mean RGB values are accumulated in a 150-frame buffer.
   - The POS rPPG projection is calculated, detrended, Butterworth band-pass
     filtered, and analyzed with an FFT.
   - The dominant frequency in 0.67--3.0 Hz is converted to beats per minute.
   - Values outside 45--160 BPM are rejected and accepted values are smoothed.

5. **Fatigue and stress indicators**
   - Eye Aspect Ratio (EAR) is computed from six landmarks per eye.
   - A blink is counted when EAR crosses below 0.18 and the debounce interval
     has elapsed.
   - Fatigue uses normalized EAR and blink rate.
   - Stress uses normalized eyebrow distance and blink rate.

6. **Jaundice and anemia indicators**
   - Sclera regions are sampled around both eyes and analyzed in YCrCb space.
   - Lip pixels are sampled and analyzed through the Cr channel.
   - A short history of 60 measurements is averaged to reduce frame noise.
   - The results are displayed as `Normal`, `Mild`, `High Risk`, or `Unknown`.

### Current limitations

- There is no labeled medical dataset or ground-truth comparison.
- Thresholds are hand-selected and are not learned from representative data.
- The camera frame rate is estimated, but the application still assumes about
  30 FPS for elapsed-time blink-rate calculation.
- Illumination normalization is limited; the white reference is a fixed
  top-right image patch and can contain arbitrary scene content.
- There is no formal confidence interval, signal-quality score, calibration
  failure state, demographic fairness analysis, or clinical sensitivity/
  specificity analysis.
- The color indicators are not sufficient to diagnose jaundice or anemia.
- The implementation does not currently store results, provide an API, or
  support longitudinal user histories.

## 4. Proposed methodology

The pipeline is intentionally staged so that unreliable frames are rejected
before physiological features are calculated:

```text
Camera frame
    -> resize and color conversion
    -> MediaPipe face landmarks
    -> face alignment and motion checks
    -> session calibration
    -> region-of-interest extraction
    -> feature extraction
    -> temporal filtering and quality checks
    -> indicator estimation
    -> uncertainty-aware display and referral message
```

### 4.1 Face landmarks and alignment

Let the normalized landmark coordinate be `(x_i, y_i)`, where both values are
in `[0, 1]`, and let the frame width and height be `W` and `H`. Pixel
coordinates are:

`X_i = x_i W`, `Y_i = y_i H`.

The guide defines an ellipse with center:

`C = (W/2, H/2)`

and radii:

`r_x = 0.15W`, `r_y = 0.25H`.

The nose displacement from the center is:

`d_x = X_nose - W/2`

`d_y = Y_nose - H/2`.

The current implementation declares the face misaligned when any of the
following conditions holds:

`face_width > 1.90 r_x` (too close),

`face_width < 1.00 r_x` (too far),

`d_x < -0.25 r_x` or `d_x > 0.25 r_x`,

`d_y < -0.25 r_y` or `d_y > 0.25 r_y`.

Otherwise the frame is aligned. Thirty consecutive aligned frames are required
before calibration begins. This reduces contamination from large pose changes
at startup; it does not prove that the face is perfectly frontal.

### 4.2 Session calibration

For each calibration frame, the left-cheek ROI is converted from BGR to HSV.
If the per-frame observations are `h_t`, `s_t`, `v_t`, green intensity `g_t`,
EAR `e_t`, and eyebrow distance `q_t`, the stored baselines are medians:

`H_0 = median(h_1, h_2, ..., h_N)`

`S_0 = median(s_1, s_2, ..., s_N)`

`V_0 = median(v_1, v_2, ..., v_N)`

`G_0 = median(g_1, g_2, ..., g_N)`

`EAR_0 = median(e_1, e_2, ..., e_N)`

`Q_0 = median(q_1, q_2, ..., q_N)`,

where `N = 90` in the current code.

The median is appropriate for a short calibration period because it is less
affected than the arithmetic mean by occasional blinks, individual noisy
frames, or short exposure changes. Calibration is personal normalization, not
medical normalization: it does not make the measurements clinically valid.

### 4.3 Eye Aspect Ratio and blink rate

For one eye, the six selected landmarks are ordered as two vertical pairs and
one horizontal pair. With Euclidean distance:

`D(P, Q) = sqrt((P_x - Q_x)^2 + (P_y - Q_y)^2)`.

The Eye Aspect Ratio is:

`EAR = [D(p_1, p_5) + D(p_2, p_4)] / [2 D(p_0, p_3)]`.

The ratio is dimensionless, so it is relatively insensitive to face distance
from the camera. The two eyes are averaged:

`EAR_avg = (EAR_left + EAR_right) / 2`.

A blink begins when:

`EAR_avg < 0.18`,

and is counted only when the previous eye-closed state is false and more than
12 frames have elapsed since the previous blink. This debounce rule prevents
one prolonged eye closure from becoming many blinks.

If `B` blinks have occurred during `T` seconds, blink rate is:

`blink_rate = 60 B / T` blinks/minute.

The present code estimates `T` from `frame_count / 30`; a production version
should use timestamps so that the measured camera rate is always respected.

The calibrated eye-opening ratio is:

`EAR_ratio = mean(EAR_history) / max(EAR_0, 0.01)`.

Current fatigue rules are:

```text
EAR_ratio < 0.70          -> High fatigue
0.70 <= EAR_ratio < 0.82  -> Moderate fatigue
EAR_ratio >= 0.82 and blink_rate < 3 -> Moderate fatigue
otherwise                 -> Low fatigue
```

These are heuristic categories and must not be interpreted as a medical
fatigue diagnosis.

### 4.4 Eyebrow elevation and stress indicator

The vector between the left and right ear landmarks is:

`v = (dx, dy) = (X_right - X_left, Y_right - Y_left)`.

Head tilt is:

`theta = atan2(dy, dx)`.

The code constructs a normal direction:

`n = (sin(theta), cos(theta))`.

For each eyebrow, the vector from eyebrow landmark to eye landmark is `u`.
The distance perpendicular to the face axis is calculated by projection:

`q = |u_x n_x + u_y n_y|`.

If `q_L` and `q_R` are the left and right values, the mean eyebrow distance is:

`q_avg = (q_L + q_R) / 2`.

Face height is the forehead-to-chin distance:

`F = |Y_chin - Y_forehead|`.

The normalized eyebrow measure is:

`eyebrow_percent = 100 q_avg / F`.

The calibrated ratio is:

`eyebrow_ratio = mean(q_history) / max(Q_0, 0.01)`.

Current stress rules combine eyebrow ratio and blink rate:

```text
eyebrow_ratio > 1.20 and blink_rate > 22 -> High stress
eyebrow_ratio > 1.12 or  blink_rate > 18 -> Moderate stress
otherwise                                  -> Low stress
```

This is a behavioral proxy, not a validated stress measurement. Eyebrow
movement can represent expression, lighting artifacts, or head pose rather
than psychological stress.

### 4.5 Remote photoplethysmography and heart rate

For each valid ROI, the application computes the spatial mean of the color
channels. Because OpenCV stores BGR, the stored signal is explicitly reordered
to:

`R_t = mean(ROI[:,:,R])`

`G_t = mean(ROI[:,:,G])`

`B_t = mean(ROI[:,:,B])`.

The signal window contains up to 150 samples. Each channel is normalized by
its window mean:

`R'_t = R_t / (mean(R) + epsilon)`

`G'_t = G_t / (mean(G) + epsilon)`

`B'_t = B_t / (mean(B) + epsilon)`,

where `epsilon = 10^-6` avoids division by zero.

The POS projection uses:

`S_1(t) = R'_t - G'_t`

`S_2(t) = R'_t + G'_t - 2B'_t`.

The adaptive projection coefficient is:

`alpha = std(S_1) / [std(S_2) + epsilon]`.

The pulse estimate is:

`p(t) = S_1(t) + alpha S_2(t)`.

The code removes a linear trend between the first and last samples:

`p_d(t) = p(t) - [p(0) + t (p(N-1)-p(0))/(N-1)]`,

then removes the remaining mean:

`p_c(t) = p_d(t) - mean(p_d)`.

This projection is useful because blood-volume changes affect facial color
subtly, while common illumination changes affect channels in a more correlated
way. POS attempts to emphasize the pulsatile component and suppress common-mode
variation. It is not a replacement for a pulse oximeter or ECG.

#### Band-pass filtering

The physiologically searched band is:

`0.67 Hz <= f <= 3.0 Hz`,

which corresponds to:

`40 BPM <= heart rate <= 180 BPM`.

The implementation designs a third-order Butterworth band-pass filter using
the current frame rate `f_s` and Nyquist frequency:

`f_N = f_s / 2`.

The normalized cutoff frequencies are:

`f_low_norm = 0.67 / f_N`

`f_high_norm = 3.0 / f_N`.

The filtered signal is obtained using zero-phase forward/backward filtering
(`filtfilt`), which reduces phase distortion. The code bypasses filtering if
the Nyquist frequency is too low for the requested upper cutoff.

#### FFT frequency selection

For a window of `N` samples and estimated sampling rate `f_s`, the real FFT
frequency bins are:

`f_k = k f_s / N`, for `k = 0, 1, ..., floor(N/2)`.

The selected peak is:

`k* = argmax_k |FFT(p_c)_k|`

subject to `0.67 <= f_k <= 3.0`.

The BPM estimate is:

`BPM = 60 f_k*`.

The application rejects estimates outside 45--160 BPM. If the new estimate
differs from the previous estimate by more than 15 BPM, it applies a
first-order update:

`BPM_new = 0.8 BPM_previous + 0.2 BPM_candidate`.

Finally, the last 15 accepted estimates are averaged:

`BPM_display = mean(BPM_history)`.

This combination of physiological band limitation, outlier rejection, and
temporal smoothing improves display stability. Correctness still requires
comparison against a reference device over controlled and uncontrolled
conditions; mathematical consistency alone cannot establish measurement
accuracy.

### 4.6 Jaundice-related sclera feature

Each sclera ROI is converted from BGR to YCrCb. For pixels whose luminance is
above 100, the mean blue-difference chroma is calculated:

`Cb_mean = mean(Cb_i | Y_i > 100)`.

The top-right image patch is used as an approximate white reference. Its
reference chroma is `Cb_white`. The illumination-corrected score is:

`Cb_corrected = Cb_mean - (Cb_white - 128)`.

The two eyes are averaged:

`J = mean(Cb_corrected_left, Cb_corrected_right)`.

The display uses a 60-sample moving average `J_smooth`. Current thresholds are:

```text
J_smooth > 135       -> High Risk
125 < J_smooth <= 135 -> Mild
J_smooth <= 125      -> Normal
```

The term “risk” is only a UI label for an image-color threshold. It is not a
diagnosis. The reference patch must be replaced with a controlled color
calibration target or a validated illumination-normalization method before
this feature can be evaluated scientifically.

### 4.7 Anemia-related lip feature

The lip ROI is converted to YCrCb. Pixels are retained when:

`Y > 60` and `Y < 200`.

The mean lip red-difference chroma is:

`Cr_mean = mean(Cr_i)`.

The normalized score is:

`A = Cr_mean / 128`.

After a 60-sample moving average `A_smooth`, current thresholds are:

```text
A_smooth < 0.90        -> High Risk
0.90 <= A_smooth < 1.05 -> Mild
A_smooth >= 1.05        -> Normal
```

The mouth-open check pauses new lip measurements when the upper-lip to
lower-lip distance exceeds 8 pixels, reducing contamination from teeth. This
feature remains an appearance heuristic and cannot infer hemoglobin
concentration without validated labels and a reference blood test.

## 5. Technology stack

| Layer | Current technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Application implementation |
| Camera/UI | OpenCV | Capture, resize, drawing, display |
| Face landmarks | MediaPipe Face Mesh | Pretrained facial landmark inference |
| Numerical processing | NumPy | Arrays, means, medians, distances, FFT inputs |
| Signal processing | SciPy | Butterworth filter and zero-phase filtering |
| Data structure | `collections.deque` | Fixed-size temporal histories |
| Deployment | Local desktop process | No account, telemetry, or hosted service |

## 6. ML model and technique assessment

### What is already used

MediaPipe Face Mesh is a pretrained machine-learning model used for landmark
estimation. The project consumes its 468/refined facial landmark output but does
not train or fine-tune that model.

The rest of the current system consists of:

- deterministic geometric feature extraction;
- color-space transformations;
- rPPG signal projection;
- Butterworth filtering and FFT peak detection;
- median/moving-average temporal smoothing;
- hand-authored threshold rules.

Therefore, the accurate description is **“a computer-vision and signal
processing screening prototype with a pretrained face-landmark model,”** not
“a trained disease-prediction model.”

### Proposed ML extension

After collecting appropriately consented, de-identified, clinically labeled
data, candidate models can be evaluated separately for each task:

1. **Heart rate regression:** ridge regression, random forest regression, or
   gradient boosting using rPPG quality and spectral features.
2. **Fatigue classification:** logistic regression, random forest, or gradient
   boosting using EAR statistics, blink rate, closure duration, and head pose.
3. **Stress classification:** calibrated logistic regression or gradient
   boosting using blink, eyebrow, motion, and temporal features.
4. **Color-screening classification:** logistic regression or gradient boosting
   using normalized YCrCb/HSV distributions, illumination features, and quality
   metrics.
5. **Temporal modeling:** only if the dataset is large enough, a 1D temporal
   convolution or recurrent model may process feature sequences. It should not
   be selected merely because it is more complex.

Any supervised model must be compared against the current rule-based baseline.
The split must be by participant, not by random frame, to prevent frames from
the same person appearing in both training and test sets.

For binary classification, the logistic model is:

`p(y=1 | x) = sigmoid(w^T x + b) = 1 / [1 + exp(-(w^T x + b))]`.

Its regularized objective can be written:

`L(w,b) = -sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)] + lambda ||w||_2^2`.

For heart-rate regression, a ridge objective is:

`L(w,b) = sum_i (y_i - (w^T x_i + b))^2 + lambda ||w||_2^2`.

These formulas define how parameters would be learned from labeled data. They
are not currently used by this repository because no training dataset exists.

## 7. Correctness, validation, and evaluation plan

Mathematical correctness means that the implementation follows its stated
definitions and handles invalid inputs. Measurement correctness requires
external evidence. The proposal therefore separates both forms of validation.

### 7.1 Software correctness

- Unit-test landmark-distance, EAR, eyebrow projection, ROI bounds, color
  conversions, median calibration, threshold boundaries, and FFT frequency
  conversion.
- Test empty ROIs, missing faces, low light, overexposure, motion, insufficient
  buffers, invalid FPS, and mouth-open frames.
- Verify that reset clears every temporal history and calibration value.
- Use timestamps for blink-rate duration and record the actual effective FPS.

### 7.2 Heart-rate validation

Record synchronized HealthScan video and a reference ECG or validated pulse
oximeter. For each recording, report:

`MAE = (1/n) sum_i |BPM_camera_i - BPM_reference_i|`

`RMSE = sqrt[(1/n) sum_i (BPM_camera_i - BPM_reference_i)^2]`.

Also report Pearson correlation, Bland–Altman bias and limits of agreement, and
the percentage of windows rejected by quality checks. Test rest, talking,
head movement, different lighting, and different camera devices.

### 7.3 Classification validation

For a clinically labeled binary outcome:

`Sensitivity = TP / (TP + FN)`

`Specificity = TN / (TN + FP)`

`Precision = TP / (TP + FP)`

`F1 = 2 Precision Sensitivity / (Precision + Sensitivity)`.

Report confidence intervals, ROC-AUC or PR-AUC as appropriate, calibration
error, and the confusion matrix. Evaluate performance by lighting, device,
skin tone, age group, sex where ethically and legally appropriate, and other
relevant subgroups. The test set must remain untouched until the final model
and thresholds are fixed.

### 7.4 Quality and uncertainty

Each output should eventually include:

- face alignment score;
- motion score;
- ROI size and brightness;
- rPPG spectral peak prominence;
- number of valid pixels;
- calibration completeness;
- an overall `Unknown` state when quality is inadequate.

An indicator should not be shown as normal merely because a measurement failed.
Unknown and low-confidence outputs are safer than success-shaped defaults.

## 8. Goals and deliverables

### Short-term goals

1. Add automated unit tests for the mathematical primitives and threshold
   boundaries.
2. Replace the fixed 30-FPS blink-time assumption with timestamp-based timing.
3. Add signal-quality indicators and explicit `Unknown`/low-confidence
   behavior.
4. Make camera source, display size, thresholds, and calibration duration
   configurable.
5. Add structured logging for debugging without recording personally
   identifiable video by default.

### Medium-term goals

1. Build a consented, de-identified, protocol-driven dataset.
2. Collect reference heart-rate measurements and clinical labels where
   applicable.
3. Compare rule-based baselines with simple interpretable supervised models.
4. Add participant-level cross-validation and subgroup analysis.
5. Calibrate probabilities and document operating thresholds.

### Long-term goals

1. Develop a validated screening workflow, not an autonomous diagnosis tool.
2. Add secure result export only after privacy and consent requirements are
   defined.
3. Establish monitoring for model drift, camera changes, and fairness.
4. Conduct appropriate clinical, regulatory, and human-factors review before
   any real-world medical claim.

## 9. Risks and mitigation

| Risk | Effect | Mitigation |
|---|---|---|
| Lighting and auto-exposure | False color and rPPG changes | Controlled illumination, reference target, quality rejection |
| Motion and pose | Noisy pulse and landmark features | Alignment gate, motion score, temporal quality checks |
| Camera/device variation | Distribution shift | Multi-device validation and per-device calibration |
| Demographic variation | Unequal error rates | Representative data and subgroup reporting |
| Medical overinterpretation | Unsafe decisions | Clear UI warnings, unknown state, professional referral |
| Data privacy | Exposure of facial video | Local processing, no default recording, consent and minimization |
| Threshold overfitting | Poor generalization | Participant-level validation and locked test set |

## 10. Conclusion

The repository already demonstrates an end-to-end local prototype: facial
landmark tracking, alignment, personal calibration, rPPG heart-rate
estimation, eye/eyebrow features, and color-based indicators are integrated
into one live application. Its strongest technical contribution at present is
the combination of a pretrained landmark detector with interpretable
signal-processing formulas and session-specific normalization.

The central next step is scientific validation. The project should not claim
that it currently predicts disease or that its thresholds are medically
correct. It should first establish reproducible signal quality and comparison
against reference measurements, then train and evaluate interpretable models
on representative, clinically labeled, participant-independent data. This
approach preserves the prototype's privacy and usability while providing a
credible path toward a reliable screening research system.

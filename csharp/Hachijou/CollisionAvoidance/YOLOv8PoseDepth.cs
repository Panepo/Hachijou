using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CollisionAvoidance
{
    public class OcclusionInfo
    {
        public bool IsOccluded { get; set; }
        public int VisibleKeypoints { get; set; }
        public List<int> MissingKeypoints { get; set; } = new List<int>();
        public string OcclusionType { get; set; } = "none";
        public int VisibleUpper { get; set; }
        public int VisibleLower { get; set; }
    }

    public class PostureInfo
    {
        public bool IsStanding { get; set; }
        public float Confidence { get; set; }
        public string Reason { get; set; } = "unknown";
        public float VerticalAlignment { get; set; }
    }

    public class BorderInfo
    {
        public bool OnBorder { get; set; }
        public List<string> Edges { get; set; } = new List<string>();
        public bool TooClose { get; set; }
    }

    public class PersonDetectionInfo
    {
        public float[] BBox { get; set; } = new float[4];
        public float[,] Keypoints { get; set; } = new float[17, 3];
        public float? DistanceCm { get; set; }
        public float? AdjustedDistanceCm { get; set; }
        public float EffectiveHeight { get; set; }
        public string HeightSource { get; set; } = "unknown";
        public float Confidence { get; set; }
        public OcclusionInfo Occlusion { get; set; } = new OcclusionInfo();
        public PostureInfo Posture { get; set; } = new PostureInfo();
        public BorderInfo Border { get; set; } = new BorderInfo();
        public float BBoxHeight { get; set; }
    }

    public partial class YOLOv8PoseDepth : IDisposable
    {
        private readonly YOLOv8PoseONNX _poseDetector;
        private readonly float _personHeightCm;
        private float? _focalLength;
        private readonly int _minKeypoints;
        private readonly int _borderThreshold;
        private readonly float _borderDistanceFactor;
        private readonly float _kptThreshold;

        public YOLOv8PoseDepth(
            string poseModelPath,
            float personHeightCm = 180.0f,
            float? focalLength = null,
            float confThreshold = 0.5f,
            float iouThreshold = 0.4f,
            int inputSize = 640,
            int minKeypoints = 10,
            int borderThreshold = 10,
            float borderDistanceFactor = 0.6f,
            float kptThreshold = 0.5f)
        {
            _poseDetector = new YOLOv8PoseONNX(
                poseModelPath,
                confThreshold,
                iouThreshold,
                new Size(inputSize, inputSize)
            );

            _personHeightCm = personHeightCm;
            _focalLength = focalLength;
            _minKeypoints = minKeypoints;
            _borderThreshold = borderThreshold;
            _borderDistanceFactor = borderDistanceFactor;
            _kptThreshold = kptThreshold;
        }

        public float EstimateFocalLength(int imageWidth, int imageHeight, float fovHorizontal = 60.0f)
        {
            float focalLengthX = (imageWidth / 2.0f) / (float)Math.Tan(fovHorizontal * Math.PI / 360.0);
            float aspectRatio = (float)imageWidth / imageHeight;
            float fovVertical = 2.0f * (float)Math.Atan(Math.Tan(fovHorizontal * Math.PI / 360.0) / aspectRatio);
            float focalLengthY = (imageHeight / 2.0f) / (float)Math.Tan(fovVertical / 2.0);
            float focalLength = (focalLengthX + focalLengthY) / 2.0f;
            return focalLength;
        }

        public float GetKeypointSpanHeight(float[,] keypoints)
        {
            var visibleY = new List<float>();

            for (int i = 0; i < 17; i++)
            {
                if (keypoints[i, 2] > _kptThreshold)
                {
                    visibleY.Add(keypoints[i, 1]);
                }
            }

            if (visibleY.Count < 2)
                return 0.0f;

            return visibleY.Max() - visibleY.Min();
        }

        public BorderInfo IsBBoxOnBorder(float[] bbox, Size frameShape)
        {
            float x1 = bbox[0];
            float y1 = bbox[1];
            float x2 = bbox[2];
            float y2 = bbox[3];

            var edges = new List<string>();

            if (x1 <= _borderThreshold)
                edges.Add("left");
            if (x2 >= frameShape.Width - _borderThreshold)
                edges.Add("right");
            if (y1 <= _borderThreshold)
                edges.Add("top");
            if (y2 >= frameShape.Height - _borderThreshold)
                edges.Add("bottom");

            return new BorderInfo
            {
                OnBorder = edges.Count > 0,
                Edges = edges
            };
        }

        public PostureInfo DetectStandingPosture(float[,] keypoints)
        {
            // Get key body points
            float[] leftShoulder = new float[] { keypoints[5, 0], keypoints[5, 1], keypoints[5, 2] };
            float[] rightShoulder = new float[] { keypoints[6, 0], keypoints[6, 1], keypoints[6, 2] };
            float[] leftHip = new float[] { keypoints[11, 0], keypoints[11, 1], keypoints[11, 2] };
            float[] rightHip = new float[] { keypoints[12, 0], keypoints[12, 1], keypoints[12, 2] };
            float[] leftKnee = new float[] { keypoints[13, 0], keypoints[13, 1], keypoints[13, 2] };
            float[] rightKnee = new float[] { keypoints[14, 0], keypoints[14, 1], keypoints[14, 2] };
            float[] leftAnkle = new float[] { keypoints[15, 0], keypoints[15, 1], keypoints[15, 2] };
            float[] rightAnkle = new float[] { keypoints[16, 0], keypoints[16, 1], keypoints[16, 2] };

            // Check visibility of critical joints
            bool shoulderVisible = leftShoulder[2] > _kptThreshold || rightShoulder[2] > _kptThreshold;
            bool hipVisible = leftHip[2] > _kptThreshold || rightHip[2] > _kptThreshold;
            bool kneeVisible = leftKnee[2] > _kptThreshold || rightKnee[2] > _kptThreshold;
            bool ankleVisible = leftAnkle[2] > _kptThreshold || rightAnkle[2] > _kptThreshold;

            if (!shoulderVisible || !hipVisible)
            {
                return new PostureInfo
                {
                    IsStanding = false,
                    Confidence = 0.0f,
                    Reason = "insufficient_keypoints",
                    VerticalAlignment = 0.0f
                };
            }

            // Calculate average positions for each body part
            float shoulderY = 0;
            int shoulderCount = 0;
            if (leftShoulder[2] > _kptThreshold) { shoulderY += leftShoulder[1]; shoulderCount++; }
            if (rightShoulder[2] > _kptThreshold) { shoulderY += rightShoulder[1]; shoulderCount++; }
            shoulderY /= Math.Max(shoulderCount, 1);

            float hipY = 0;
            int hipCount = 0;
            if (leftHip[2] > _kptThreshold) { hipY += leftHip[1]; hipCount++; }
            if (rightHip[2] > _kptThreshold) { hipY += rightHip[1]; hipCount++; }
            hipY /= Math.Max(hipCount, 1);

            float kneeY = 0;
            int kneeCount = 0;
            if (leftKnee[2] > _kptThreshold) { kneeY += leftKnee[1]; kneeCount++; }
            if (rightKnee[2] > _kptThreshold) { kneeY += rightKnee[1]; kneeCount++; }
            kneeY /= Math.Max(kneeCount, 1);

            float ankleY = 0;
            int ankleCount = 0;
            if (leftAnkle[2] > _kptThreshold) { ankleY += leftAnkle[1]; ankleCount++; }
            if (rightAnkle[2] > _kptThreshold) { ankleY += rightAnkle[1]; ankleCount++; }
            ankleY /= Math.Max(ankleCount, 1);

            // Calculate torso height (shoulder to hip)
            float torsoHeight = Math.Abs(hipY - shoulderY);

            // Calculate vertical alignment metrics
            float verticalAlignment = 0.0f;

            if (kneeVisible && ankleVisible && torsoHeight > 0)
            {
                // Full body visible
                float hipKneeHeight = Math.Abs(kneeY - hipY);
                float kneeAnkleHeight = Math.Abs(ankleY - kneeY);

                bool shoulderToHipOk = hipY > shoulderY;
                bool hipToKneeOk = kneeY > hipY;
                bool kneeToAnkleOk = ankleY > kneeY;

                float totalHeight = ankleY - shoulderY;

                if (totalHeight > 0)
                {
                    float torsoRatio = torsoHeight / totalHeight;
                    float legRatio = (hipKneeHeight + kneeAnkleHeight) / totalHeight;

                    bool isProportional = (0.30f < torsoRatio && torsoRatio < 0.55f) &&
                                         (0.45f < legRatio && legRatio < 0.70f);

                    var alignmentChecks = new[] { shoulderToHipOk, hipToKneeOk, kneeToAnkleOk, isProportional };
                    verticalAlignment = alignmentChecks.Count(x => x) / (float)alignmentChecks.Length;

                    if (verticalAlignment >= 0.80f)
                    {
                        return new PostureInfo
                        {
                            IsStanding = true,
                            Confidence = verticalAlignment,
                            Reason = "vertical_full_body",
                            VerticalAlignment = verticalAlignment
                        };
                    }
                    else
                    {
                        return new PostureInfo
                        {
                            IsStanding = false,
                            Confidence = 1.0f - verticalAlignment,
                            Reason = "sitting_or_lying",
                            VerticalAlignment = verticalAlignment
                        };
                    }
                }
            }
            else if (kneeVisible && torsoHeight > 0)
            {
                // Knees visible but not ankles
                float hipKneeHeight = Math.Abs(kneeY - hipY);

                bool shoulderToHipOk = hipY > shoulderY;
                bool hipToKneeOk = kneeY > hipY;

                float upperBodyHeight = kneeY - shoulderY;
                if (upperBodyHeight > 0)
                {
                    float torsoRatio = torsoHeight / upperBodyHeight;

                    bool isProportional = 0.35f < torsoRatio && torsoRatio < 0.65f;

                    var alignmentChecks = new[] { shoulderToHipOk, hipToKneeOk, isProportional };
                    verticalAlignment = alignmentChecks.Count(x => x) / (float)alignmentChecks.Length;

                    if (verticalAlignment >= 0.66f)
                    {
                        return new PostureInfo
                        {
                            IsStanding = true,
                            Confidence = verticalAlignment * 0.8f,
                            Reason = "vertical_upper_body",
                            VerticalAlignment = verticalAlignment
                        };
                    }
                }
            }

            return new PostureInfo
            {
                IsStanding = false,
                Confidence = 0.6f,
                Reason = "unclear_posture",
                VerticalAlignment = verticalAlignment
            };
        }

        public OcclusionInfo DetectOcclusion(float[,] keypoints)
        {
            // Define keypoint groups
            int[] upperBody = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] lowerBody = { 11, 12, 13, 14, 15, 16 };

            var visibleKeypoints = new List<int>();
            var missingKeypoints = new List<int>();
            int visibleUpper = 0;
            int visibleLower = 0;

            for (int i = 0; i < 17; i++)
            {
                if (keypoints[i, 2] > _kptThreshold)
                {
                    visibleKeypoints.Add(i);
                    if (upperBody.Contains(i))
                        visibleUpper++;
                    if (lowerBody.Contains(i))
                        visibleLower++;
                }
                else
                {
                    missingKeypoints.Add(i);
                }
            }

            int numVisible = visibleKeypoints.Count;
            bool isOccluded = false;
            string occlusionType = "none";

            if (numVisible < 10)
            {
                isOccluded = true;
                occlusionType = "partial";
            }
            else if (visibleLower < 4 && visibleUpper >= 3)
            {
                isOccluded = true;
                occlusionType = "lower_body";
            }
            else if (visibleUpper < 5 && visibleLower >= 2)
            {
                isOccluded = true;
                occlusionType = "upper_body";
            }

            return new OcclusionInfo
            {
                IsOccluded = isOccluded,
                VisibleKeypoints = numVisible,
                MissingKeypoints = missingKeypoints,
                OcclusionType = occlusionType,
                VisibleUpper = visibleUpper,
                VisibleLower = visibleLower
            };
        }

        public (float? distanceCm, float effectiveHeight, string heightSource) CalculateDistanceFromBBox(
            float[] bbox,
            OcclusionInfo? occlusionInfo = null,
            PostureInfo? postureInfo = null)
        {
            if (_focalLength == null || _focalLength <= 0)
                return (null, _personHeightCm, "unknown");

            float x1 = bbox[0];
            float y1 = bbox[1];
            float x2 = bbox[2];
            float y2 = bbox[3];
            float pixelHeight = y2 - y1;

            if (pixelHeight <= 0)
                return (null, _personHeightCm, "unknown");

            // Adjust person height based on posture and occlusion
            float effectiveHeight = _personHeightCm;
            string heightSource = "full_standing";

            // Check posture first
            if (postureInfo != null && postureInfo.Confidence > 0.5f)
            {
                if (!postureInfo.IsStanding)
                {
                    effectiveHeight = _personHeightCm * 0.60f;
                    heightSource = "sitting_fixed";
                }
            }
            // Fall back to occlusion heuristics
            else if (occlusionInfo != null)
            {
                float bboxWidth = x2 - x1;
                float bboxHeight = y2 - y1;
                float aspectRatio = bboxHeight > 0 ? bboxWidth / bboxHeight : 0;
                bool isNearSquare = 0.8f <= aspectRatio && aspectRatio <= 1.2f;

                if (isNearSquare && occlusionInfo.OcclusionType == "partial")
                {
                    effectiveHeight = _personHeightCm / 3.0f;
                    heightSource = "crouched";
                }
                else if (occlusionInfo.OcclusionType == "partial")
                {
                    effectiveHeight = _personHeightCm / 2.5f;
                    heightSource = "partial_occluded";
                }
                else if (occlusionInfo.OcclusionType == "lower_body")
                {
                    effectiveHeight = _personHeightCm / 2.0f;
                    heightSource = "upper_body_only";
                }
                else if (occlusionInfo.OcclusionType == "upper_body")
                {
                    effectiveHeight = _personHeightCm / 2.0f;
                    heightSource = "lower_body_only";
                }
            }

            float distance = (effectiveHeight * _focalLength.Value) / pixelHeight;
            return (distance, effectiveHeight, heightSource);
        }

        public float? CalibrateFocalLength(float[] bbox, float knownDistanceCm)
        {
            float x1 = bbox[0];
            float y1 = bbox[1];
            float x2 = bbox[2];
            float y2 = bbox[3];
            float pixelHeight = y2 - y1;

            if (pixelHeight <= 0)
                return null;

            float focalLength = (pixelHeight * knownDistanceCm) / _personHeightCm;
            return focalLength;
        }

        public Mat DrawPersonWithInfo(
            Mat image,
            Detection detection,
            float? distanceCm,
            float effectiveHeight,
            string heightSource,
            OcclusionInfo occlusionInfo,
            PostureInfo postureInfo,
            BorderInfo borderInfo,
            bool drawKeypoints = true,
            bool drawSkeleton = true)
        {
            float[] bbox = detection.BBox;
            float confidence = detection.Confidence;
            float[,] keypoints = detection.Keypoints;

            int x1 = (int)bbox[0];
            int y1 = (int)bbox[1];
            int x2 = (int)bbox[2];
            int y2 = (int)bbox[3];

            // Choose color based on proximity and occlusion
            Scalar boxColor;
            if (borderInfo.TooClose)
            {
                boxColor = new Scalar(0, 0, 255); // Red
            }
            else if (borderInfo.OnBorder)
            {
                boxColor = new Scalar(0, 100, 255); // Orange-red
            }
            else if (occlusionInfo.IsOccluded)
            {
                boxColor = new Scalar(0, 165, 255); // Orange
            }
            else if (postureInfo != null && !postureInfo.IsStanding && postureInfo.Confidence > 0.5f)
            {
                boxColor = new Scalar(255, 165, 0); // Blue
            }
            else
            {
                boxColor = new Scalar(0, 255, 0); // Green
            }

            // Draw bounding box
            Cv2.Rectangle(image, new Point(x1, y1), new Point(x2, y2), boxColor, 2);

            // Draw skeleton
            if (drawSkeleton)
            {
                foreach (var sk in _poseDetector.Skeleton)
                {
                    int kpt1Idx = sk[0] - 1;
                    int kpt2Idx = sk[1] - 1;

                    if (kpt1Idx >= 17 || kpt2Idx >= 17)
                        continue;

                    float kpt1Conf = keypoints[kpt1Idx, 2];
                    float kpt2Conf = keypoints[kpt2Idx, 2];

                    if (kpt1Conf > _kptThreshold && kpt2Conf > _kptThreshold)
                    {
                        Point pt1 = new Point((int)keypoints[kpt1Idx, 0], (int)keypoints[kpt1Idx, 1]);
                        Point pt2 = new Point((int)keypoints[kpt2Idx, 0], (int)keypoints[kpt2Idx, 1]);
                        Cv2.Line(image, pt1, pt2, new Scalar(255, 150, 0), 2);
                    }
                }
            }

            // Draw keypoints
            if (drawKeypoints)
            {
                for (int i = 0; i < 17; i++)
                {
                    float kptConf = keypoints[i, 2];
                    if (kptConf > _kptThreshold)
                    {
                        Point pt = new Point((int)keypoints[i, 0], (int)keypoints[i, 1]);
                        Cv2.Circle(image, pt, 4, new Scalar(0, 255, 0), -1);
                        Cv2.Circle(image, pt, 5, Scalar.Black, 1);
                    }
                    else if (kptConf > 0.1f)
                    {
                        Point pt = new Point((int)keypoints[i, 0], (int)keypoints[i, 1]);
                        Cv2.Circle(image, pt, 3, new Scalar(100, 100, 100), -1);
                    }
                }
            }

            // Prepare labels
            var labels = new List<string>();
            labels.Add($"Person: {confidence:F2}");
            labels.Add($"Keypoints: {occlusionInfo.VisibleKeypoints}/17");

            // Add posture information
            if (postureInfo != null && postureInfo.Confidence > 0.3f)
            {
                if (postureInfo.IsStanding)
                {
                    labels.Add($"Standing ({postureInfo.Confidence:P0})");
                }
                else
                {
                    string postureLabel = postureInfo.Reason == "sitting_or_lying" ? "Sitting" : "Not Standing";
                    labels.Add($"{postureLabel} ({postureInfo.Confidence:P0})");
                }
            }

            if (distanceCm != null)
            {
                string distStr;
                if (distanceCm >= 100)
                {
                    distStr = $"{distanceCm / 100:F2}m";
                }
                else
                {
                    distStr = $"{distanceCm:F0}cm";
                }

                if (heightSource == "sitting_fixed")
                {
                    labels.Add($"Dist: {distStr} (h={effectiveHeight:F0}cm)");
                }
                else
                {
                    labels.Add($"Distance: {distStr}");
                }
            }
            else
            {
                labels.Add("Distance: N/A");
            }

            if (borderInfo.TooClose)
            {
                labels.Add("TOO CLOSE! (border+few kpts)");
            }
            else if (borderInfo.OnBorder)
            {
                string edgesStr = string.Join(",", borderInfo.Edges);
                labels.Add($"Very Close! ({edgesStr})");
            }
            else if (occlusionInfo.IsOccluded)
            {
                if (occlusionInfo.OcclusionType == "lower_body" || occlusionInfo.OcclusionType == "upper_body")
                {
                    labels.Add($"Half body ({occlusionInfo.OcclusionType})");
                }
                else
                {
                    labels.Add($"Occluded: {occlusionInfo.OcclusionType}");
                }
            }

            // Calculate label background size
            var textSizes = labels.Select(lbl =>
            {
                Size size = Cv2.GetTextSize(lbl, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
                return (size, baseline);
            }).ToList();

            int maxWidth = textSizes.Max(t => t.size.Width);
            int totalHeight = textSizes.Sum(t => t.size.Height + t.baseline) + 10;

            // Draw label background
            Cv2.Rectangle(image,
                new Point(x1, y1 - totalHeight - 5),
                new Point(x1 + maxWidth + 10, y1),
                boxColor, -1);

            // Draw labels
            int yOffset = y1 - 5;
            for (int i = labels.Count - 1; i >= 0; i--)
            {
                Cv2.PutText(image, labels[i], new Point(x1 + 5, yOffset),
                    HersheyFonts.HersheySimplex, 0.5, Scalar.White, 1, LineTypes.AntiAlias);
                yOffset -= textSizes[i].size.Height + textSizes[i].baseline + 2;
            }

            return image;
        }

        public (Mat outputImage, int numPersons, List<PersonDetectionInfo> detectionsInfo) ProcessFrame(
            Mat frame,
            bool drawKeypoints = true,
            bool drawSkeleton = true)
        {
            // Run pose detection
            var detections = _poseDetector.Detect(frame);

            // Create output image
            Mat output = frame.Clone();
            var personDetections = new List<PersonDetectionInfo>();

            // Process each detected person
            foreach (var detection in detections)
            {
                float[] bbox = detection.BBox;
                float[,] keypoints = detection.Keypoints;

                // Border detection
                BorderInfo borderInfo = IsBBoxOnBorder(bbox, new Size(frame.Width, frame.Height));

                // Occlusion detection
                OcclusionInfo occlusionInfo = DetectOcclusion(keypoints);

                // Posture detection
                PostureInfo postureInfo = DetectStandingPosture(keypoints);

                // Distance calculation
                var (distanceCm, effectiveHeight, heightSource) = CalculateDistanceFromBBox(bbox, occlusionInfo, postureInfo);

                // Adjust distance for border cases
                float? adjustedDistanceCm = distanceCm;
                bool isTooClose = false;

                if (distanceCm != null)
                {
                    if (borderInfo.OnBorder && occlusionInfo.VisibleKeypoints < _minKeypoints)
                    {
                        isTooClose = true;
                        adjustedDistanceCm = distanceCm * _borderDistanceFactor;
                    }
                }

                borderInfo.TooClose = isTooClose;

                // Draw annotations
                output = DrawPersonWithInfo(
                    output, detection, adjustedDistanceCm, effectiveHeight, heightSource,
                    occlusionInfo, postureInfo, borderInfo,
                    drawKeypoints, drawSkeleton
                );

                // Store detection info
                personDetections.Add(new PersonDetectionInfo
                {
                    BBox = bbox,
                    Keypoints = keypoints,
                    DistanceCm = distanceCm,
                    AdjustedDistanceCm = adjustedDistanceCm,
                    EffectiveHeight = effectiveHeight,
                    HeightSource = heightSource,
                    Confidence = detection.Confidence,
                    Occlusion = occlusionInfo,
                    Posture = postureInfo,
                    Border = borderInfo,
                    BBoxHeight = bbox[3] - bbox[1]
                });
            }

            return (output, personDetections.Count, personDetections);
        }

        public void SetFocalLength(float focalLength)
        {
            _focalLength = focalLength;
        }

        public float? GetFocalLength()
        {
            return _focalLength;
        }

        public void Dispose()
        {
            _poseDetector?.Dispose();
        }
    }
}

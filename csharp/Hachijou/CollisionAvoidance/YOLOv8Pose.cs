using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CollisionAvoidance
{
    public class Detection
    {
        public float[] BBox { get; set; } = new float[4]; // [x1, y1, x2, y2]
        public float Confidence { get; set; }
        public float[,] Keypoints { get; set; } = new float[17, 3]; // [17, 3] for x, y, conf
    }

    public partial class YOLOv8PoseONNX : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _modelPath;
        private readonly float _confThreshold;
        private readonly float _iouThreshold;
        private readonly Size _inputSize;

        public string[] KeypointNames { get; } = new string[]
        {
            "nose",           // 0
            "left_eye",       // 1
            "right_eye",      // 2
            "left_ear",       // 3
            "right_ear",      // 4
            "left_shoulder",  // 5
            "right_shoulder", // 6
            "left_elbow",     // 7
            "right_elbow",    // 8
            "left_wrist",     // 9
            "right_wrist",    // 10
            "left_hip",       // 11
            "right_hip",      // 12
            "left_knee",      // 13
            "right_knee",     // 14
            "left_ankle",     // 15
            "right_ankle"     // 16
        };

        public int[][] Skeleton { get; } = new int[][]
        {
            new int[] {16, 14}, new int[] {14, 12}, new int[] {17, 15}, new int[] {15, 13}, new int[] {12, 13}, // legs
            new int[] {6, 12}, new int[] {7, 13},  // torso
            new int[] {6, 8}, new int[] {7, 9}, new int[] {8, 10}, new int[] {9, 11},  // arms
            new int[] {2, 3}, new int[] {1, 2}, new int[] {1, 3}, new int[] {2, 4}, new int[] {3, 5}, new int[] {4, 6}, new int[] {5, 7}  // face to shoulders
        };

        public Scalar KptColor { get; set; } = new Scalar(0, 255, 0);  // Green for keypoints
        public Scalar SkeletonColor { get; set; } = new Scalar(255, 100, 0);  // Blue for skeleton
        public Scalar BoxColor { get; set; } = new Scalar(255, 0, 255);  // Magenta

        public YOLOv8PoseONNX(string modelPath, float confThreshold = 0.5f, float iouThreshold = 0.4f, Size? inputSize = null)
        {
            _modelPath = modelPath;
            _confThreshold = confThreshold;
            _iouThreshold = iouThreshold;
            _inputSize = inputSize ?? new Size(640, 640);

            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA();
            sessionOptions.AppendExecutionProvider_CPU();

            _session = new InferenceSession(modelPath, sessionOptions);

            var inputNames = string.Join(", ", _session.InputMetadata.Keys);
            var outputNames = string.Join(", ", _session.OutputMetadata.Keys);

            Console.WriteLine($"[YOLOv8-Pose] Model loaded: {modelPath}");
            Console.WriteLine($"[YOLOv8-Pose] Input names: {inputNames}");
            Console.WriteLine($"[YOLOv8-Pose] Output names: {outputNames}");
            Console.WriteLine($"[YOLOv8-Pose] Input size: {_inputSize}");
        }

        private (DenseTensor<float> tensor, float scale, int padW, int padH) Preprocess(Mat image)
        {
            int height = image.Height;
            int width = image.Width;

            float scale = Math.Min((float)_inputSize.Height / height, (float)_inputSize.Width / width);
            int newHeight = (int)(height * scale);
            int newWidth = (int)(width * scale);

            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(newWidth, newHeight), interpolation: InterpolationFlags.Linear);

            Mat canvas = new Mat(_inputSize.Height, _inputSize.Width, MatType.CV_8UC3, new Scalar(114, 114, 114));

            int padH = (_inputSize.Height - newHeight) / 2;
            int padW = (_inputSize.Width - newWidth) / 2;

            Mat roi = new Mat(canvas, new Rect(padW, padH, newWidth, newHeight));
            resized.CopyTo(roi);

            Cv2.CvtColor(canvas, canvas, ColorConversionCodes.BGR2RGB);

            var tensor = new DenseTensor<float>(new[] { 1, 3, _inputSize.Height, _inputSize.Width });

            for (int y = 0; y < _inputSize.Height; y++)
            {
                for (int x = 0; x < _inputSize.Width; x++)
                {
                    Vec3b pixel = canvas.At<Vec3b>(y, x);
                    tensor[0, 0, y, x] = pixel[0] / 255.0f;
                    tensor[0, 1, y, x] = pixel[1] / 255.0f;
                    tensor[0, 2, y, x] = pixel[2] / 255.0f;
                }
            }

            resized.Dispose();
            canvas.Dispose();
            roi.Dispose();

            return (tensor, scale, padW, padH);
        }

        private List<Detection> Postprocess(IReadOnlyList<DisposableNamedOnnxValue> outputs, Size originalShape, float scale, int padW, int padH)
        {
            var predictions = outputs[0].AsEnumerable<float>().ToArray();
            var outputShape = outputs[0].AsTensor<float>().Dimensions.ToArray();

            int numDetections = outputShape[2]; // 8400
            int numChannels = outputShape[1];   // 56

            var detections = new List<Detection>();
            int origHeight = originalShape.Height;
            int origWidth = originalShape.Width;

            for (int i = 0; i < numDetections; i++)
            {
                int baseIdx = i * numChannels;

                float xCenter = predictions[baseIdx + 0];
                float yCenter = predictions[baseIdx + 1];
                float w = predictions[baseIdx + 2];
                float h = predictions[baseIdx + 3];
                float confidence = predictions[baseIdx + 4];

                if (confidence < _confThreshold)
                    continue;

                float x1 = xCenter - w / 2;
                float y1 = yCenter - h / 2;
                float x2 = xCenter + w / 2;
                float y2 = yCenter + h / 2;

                x1 = (x1 - padW) / scale;
                y1 = (y1 - padH) / scale;
                x2 = (x2 - padW) / scale;
                y2 = (y2 - padH) / scale;

                x1 = Math.Max(0, Math.Min(x1, origWidth));
                y1 = Math.Max(0, Math.Min(y1, origHeight));
                x2 = Math.Max(0, Math.Min(x2, origWidth));
                y2 = Math.Max(0, Math.Min(y2, origHeight));

                if (x2 <= x1 || y2 <= y1)
                    continue;

                var keypoints = new float[17, 3];
                for (int j = 0; j < 17; j++)
                {
                    float kptX = predictions[baseIdx + 5 + j * 3];
                    float kptY = predictions[baseIdx + 5 + j * 3 + 1];
                    float kptConf = predictions[baseIdx + 5 + j * 3 + 2];

                    kptX = (kptX - padW) / scale;
                    kptY = (kptY - padH) / scale;

                    keypoints[j, 0] = kptX;
                    keypoints[j, 1] = kptY;
                    keypoints[j, 2] = kptConf;
                }

                detections.Add(new Detection
                {
                    BBox = new float[] { x1, y1, x2, y2 },
                    Confidence = confidence,
                    Keypoints = keypoints
                });
            }

            if (detections.Count > 0)
            {
                detections = ApplyNMS(detections);
            }

            return detections;
        }

        private List<Detection> ApplyNMS(List<Detection> detections)
        {
            if (detections.Count == 0)
                return new List<Detection>();

            var rects = detections.Select(d => new Rect(
                (int)d.BBox[0],
                (int)d.BBox[1],
                (int)(d.BBox[2] - d.BBox[0]),
                (int)(d.BBox[3] - d.BBox[1])
            )).ToArray();

            var confidences = detections.Select(d => d.Confidence).ToArray();

            CvDnn.NMSBoxes(rects, confidences, _confThreshold, _iouThreshold, out int[] indices);

            return indices.Select(i => detections[i]).ToList();
        }

        public List<Detection> Detect(Mat image)
        {
            var (inputTensor, scale, padW, padH) = Preprocess(image);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
            };

            using var outputs = _session.Run(inputs);
            var detections = Postprocess(outputs.ToList(), new Size(image.Width, image.Height), scale, padW, padH);

            return detections;
        }

        public Mat DrawDetections(Mat image, List<Detection> detections, bool drawBBox = true,
                                  bool drawKeypoints = true, bool drawSkeleton = true, float kptThreshold = 0.5f)
        {
            Mat outputImage = image.Clone();

            foreach (var det in detections)
            {
                int x1 = (int)det.BBox[0];
                int y1 = (int)det.BBox[1];
                int x2 = (int)det.BBox[2];
                int y2 = (int)det.BBox[3];

                if (drawBBox)
                {
                    Cv2.Rectangle(outputImage, new Point(x1, y1), new Point(x2, y2), BoxColor, 2);

                    string label = $"Person: {det.Confidence:F2}";
                    Size labelSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 1, out int baseline);

                    Cv2.Rectangle(outputImage,
                        new Point(x1, y1 - labelSize.Height - baseline - 5),
                        new Point(x1 + labelSize.Width, y1),
                        BoxColor, -1);

                    Cv2.PutText(outputImage, label, new Point(x1, y1 - baseline - 2),
                        HersheyFonts.HersheySimplex, 0.6, Scalar.White, 1);
                }

                if (drawSkeleton)
                {
                    foreach (var sk in Skeleton)
                    {
                        int kpt1Idx = sk[0] - 1;
                        int kpt2Idx = sk[1] - 1;

                        if (kpt1Idx >= 17 || kpt2Idx >= 17)
                            continue;

                        float kpt1Conf = det.Keypoints[kpt1Idx, 2];
                        float kpt2Conf = det.Keypoints[kpt2Idx, 2];

                        if (kpt1Conf > kptThreshold && kpt2Conf > kptThreshold)
                        {
                            Point pt1 = new Point((int)det.Keypoints[kpt1Idx, 0], (int)det.Keypoints[kpt1Idx, 1]);
                            Point pt2 = new Point((int)det.Keypoints[kpt2Idx, 0], (int)det.Keypoints[kpt2Idx, 1]);
                            Cv2.Line(outputImage, pt1, pt2, SkeletonColor, 2);
                        }
                    }
                }

                if (drawKeypoints)
                {
                    for (int i = 0; i < 17; i++)
                    {
                        float kptConf = det.Keypoints[i, 2];
                        if (kptConf > kptThreshold)
                        {
                            Point pt = new Point((int)det.Keypoints[i, 0], (int)det.Keypoints[i, 1]);
                            Cv2.Circle(outputImage, pt, 4, KptColor, -1);
                            Cv2.Circle(outputImage, pt, 5, Scalar.Black, 1);
                        }
                    }
                }
            }

            return outputImage;
        }

        public (List<Detection> detections, Mat annotatedImage) DetectAndDraw(Mat image, bool drawBBox = true,
                                                                                bool drawKeypoints = true,
                                                                                bool drawSkeleton = true,
                                                                                float kptThreshold = 0.5f)
        {
            var detections = Detect(image);
            var annotatedImage = DrawDetections(image, detections, drawBBox, drawKeypoints, drawSkeleton, kptThreshold);
            return (detections, annotatedImage);
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

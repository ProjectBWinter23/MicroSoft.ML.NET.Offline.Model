using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;


namespace ImageRecognition
{
    public static class Accessor
    {
        private static List<string> classes = new List<string>() { "cardboard", "glass", "metal", "paper", "plastic", "trash" };

        public static async void Predict_LocalModel(string picturePath, string modelPath)
        {
            // Load ml model
            var mlCtx2 = new MLContext();
            var loadedModel = mlCtx2.Model.Load(modelPath, out var _);
            var predictionEngine = mlCtx2.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadedModel);

            // Predict 
            await using var imagesStream = File.Open(picturePath, FileMode.Open);
            ModelOutput prediction = predictionEngine.Predict(new ModelInput(imagesStream));
            float score = prediction.Prediction.Max();
            int index = Array.IndexOf(prediction.Prediction, score);
             
            string result = "This image most likely belongs to " + classes[index] + " with a " + " percent confidence." + Math.Round(score,2);
        }

        public static void SaveAsMicrosoftMLModel()
        {
            // Configure ML model
            var mlCtx = new MLContext();


            var pipeline = mlCtx
                .Transforms
                // Adjust the image to the required model input size
                .ResizeImages(
                    inputColumnName: nameof(ModelInput.Image),
                    imageWidth: ModelInput.ImageWidth,
                    imageHeight: ModelInput.ImageHeight,
                    outputColumnName: "resized"
                )
                // Extract the pixels form the image as a 1D float array, but keep them in the same order as they appear in the image.
                .Append(mlCtx.Transforms.ExtractPixels(
                    inputColumnName: "resized",
                    interleavePixelColors: true,
                    outputAsFloatArray: true,
                    outputColumnName: LandmarkModelSettings.Input)
                )
                // Perform the estimation
                .Append(mlCtx.Transforms.ApplyOnnxModel(
                        modelFile: LandmarkModelSettings.OnnxModelName,
                        inputColumnName: LandmarkModelSettings.Input,
                        outputColumnName: LandmarkModelSettings.Output
                    )
                );

            // Save ml model
            var transformer = pipeline.Fit(mlCtx.Data.LoadFromEnumerable(new List<ModelInput>()));

            mlCtx.Model.Save(transformer, null, LandmarkModelSettings.MlNetModelFileName);
        }
    }

    public static class LandmarkModelSettings
    {
        public const string OnnxModelName = "model3.onnx";
        public const string Input = "serving_default_sequential_input:0";
        public const string Output = "StatefulPartitionedCall:0";

        public const string MlNetModelFileName = "D:\\RheinWaal_Msc\\Applied_Project_B\\RecyclingIsFun\\ImageRecognition\\local_waste_classifier_onnx.zip";
        public const string LabelFileName = "landmarks_classifier_north_america_V1_label_map.csv";
    }

    public class ModelInput
    {
        public const int ImageWidth = 256;
        public const int ImageHeight = 256;

        public ModelInput(Stream imagesStream)
        {
            Image = MLImage.CreateFromStream(imagesStream);
        }

        [ImageType(width: ImageWidth, height: ImageHeight)]
        public MLImage Image { get; }
    }

    public class ModelOutput
    {
        [ColumnName(LandmarkModelSettings.Output)]
        public float[] Prediction { get; set; }
    }
}

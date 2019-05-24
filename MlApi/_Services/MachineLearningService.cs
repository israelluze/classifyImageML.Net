using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Image;

namespace MlApi._Services
{

    public class MachineLearningService
    {
        //setar variáveis utilizadas 
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        //esta é o local onde estão as tags customizadas
        static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
        //pasta onde ficam as imagens que correspondem as tags e que serão utilizadas para treinar o modelo
        static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
        //variável que receberá a imagem que desejamos classificar
        private string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "yodaSextou.jpg");
        //define o modelo tensorflow que será utilizado na transferencia de treinamento
        static readonly string _inceptionPb = Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        //local de armazenado do modelo depois de treinado
        static readonly string _outputImageClassifierZip = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");
        private static string LabelTokey = nameof(LabelTokey);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        public ImagePrediction ClassificarImagem(string nomeImagem)
        {

            _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", nomeImagem);

            MLContext mlContext = new MLContext(seed: 1);
            //treinamento do modelo
            var model = ReuseAndTuneInceptionModel(mlContext, _trainTagsTsv, _trainImagesFolder, _inceptionPb, _outputImageClassifierZip);
            //efetua a classificação da imagem
            return ClassifySingleImage(mlContext, _predictSingleImage, _outputImageClassifierZip, model);           


        }
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }
        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
                     .Select(line => line.Split('\t'))
                     .Select(line => new ImageData()
                     {
                         ImagePath = Path.Combine(folder, line[0])
                     });

        }

        public static ITransformer ReuseAndTuneInceptionModel(MLContext mlContext, string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            var data = mlContext.Data.LoadFromTextFile<ImageData>(path: dataLocation, hasHeader: false);

            var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation).
                    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            ITransformer model = estimator.Fit(data);

            var predictions = model.Transform(data);

            var imageData = mlContext.Data.CreateEnumerable<ImageData>(data, false, true);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);


            Console.WriteLine("=============== Classification metrics ===============");

            var multiclassContext = mlContext.MulticlassClassification;
            var metrics = multiclassContext.Evaluate(predictions, labelColumnName: LabelTokey, predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;


        }

        public static ImagePrediction ClassifySingleImage(MLContext mlContext, string imagePath, string outputModelLocation, ITransformer model)
        {

            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };

            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);
            prediction.ImagePath = Path.GetFileName(imageData.ImagePath); 
            prediction.ScoreMax = prediction.Score.Max();

            Console.WriteLine("=============== Making single image classification ===============");            
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");


            return prediction;
        }

    }
}
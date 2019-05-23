using System;

namespace MlApi._Services
{
    public class ImagePrediction
    {
        public float[] Score;

        public string PredictedLabelValue;      

        public string ImagePath { get; set; } 

        public float ScoreMax { get; set; }
        
    }
}
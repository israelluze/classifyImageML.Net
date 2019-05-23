using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using MlApi._Services;

namespace MlApi.Controllers
{
    [Route("api/[controller]")]    
    public class MlController : Controller
    {
        [HttpGet("{nomeImagem}")]
        public ImagePrediction Get(string nomeImagem) {            

            var retorno = new MachineLearningService().ClassificarImagem(nomeImagem);

            return retorno; // 

        }
    }
}
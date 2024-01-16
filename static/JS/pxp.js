
function extractText() {
  var fileInput = document.getElementById('file');
  var file = fileInput.files[0];

  if (!file) {
      var output = document.getElementById('output');
      output.innerHTML = '<label class="error">seleccione un archivo</label>';
      return false;
  }else if (!file.name.match(/\.(xlsx|csv)$/i)) {
      var output = document.getElementById('output');
      output.innerHTML = '<label class="error">seleccione archivo .csv.</label>';
      return false;
  } else if (file.size == 0) {
      var output = document.getElementById('output');
      output.innerHTML = '<label class="error">El archivo está vacío</label>';
      return false;
  }

  // Add any additional validation logic here...

  return true;
}

function displayFileName(input) {
  var preview = document.getElementById('preview');
  if (input.files.length > 0) {
      preview.innerHTML = 'Archivo seleccionado: ' + input.files[0].name;
  } else {
      preview.innerHTML = 'Selecciona Archivo';
  }
}

function handleDragOver(event) {
    event.preventDefault();
    var fileLabel = document.getElementById('file-label');
    fileLabel.style.backgroundColor = '#f2f2f2';
}

function handleFileSelect(event) {
    event.preventDefault();
    var fileLabel = document.getElementById('file-label');
    fileLabel.style.backgroundColor = '';

    var files = event.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('file').files = files;
        displayFileName(document.getElementById('file'));
    }
}

function displayFileName(input) {
    var preview = document.getElementById('preview');
    if (input.files.length > 0) {
        preview.innerHTML = input.files[0].name;
    } else {
        preview.innerHTML = 'Selecciona o Arrastra el Archivo';
    }
}

//graficas

document.addEventListener("DOMContentLoaded", function() {
  // Obtén el elemento canvas y el valor del atributo 'data-ingreso'
  var canvas = document.getElementById('porcentajeChart');
  var ingreso = parseFloat(canvas.getAttribute('data-ingreso'));

  // Redondea el número a dos decimales
  ingresow = parseFloat(ingreso.toFixed(2));

  // Calcula el porcentaje restante
  var porcentajeRestante = 100 - ingresow;

  // Configuración de la gráfica
  var config = {
      type: 'doughnut',
      data: {
          datasets: [{
              data: [ingresow, porcentajeRestante],
              backgroundColor: [
                  'rgba(51, 196, 129)', // Color para el valor de ingreso
                  'rgba(150, 42, 68)' // Color para el porcentaje restante (fondo blanco)
              ],
              borderWidth: 1
          }]
      },
      options: {
          cutoutPercentage: 100, // Porcentaje de recorte para crear el efecto de dona
          responsive: false
      }
  };

  // Crear la instancia de la gráfica
  var porcentajeChart = new Chart(canvas, config);
});

//ventas graficas





//ventanas
var btnAbrirPopup = document.getElementById('btn-abrir-popup'),
overlay = document.getElementById('overlay'),
popup = document.getElementById('popup'),
btnCerrarPopup = document.getElementById('btn-cerrar-popup');

btnAbrirPopup.addEventListener('click', function(){
overlay.classList.add('active');
popup.classList.add('active');
});

btnCerrarPopup.addEventListener('click', function(e){
e.preventDefault();
overlay.classList.remove('active');
popup.classList.remove('active');
});
var btnAbrirPopup1 = document.getElementById('btn-abrir-popup1'),
	overlay1 = document.getElementById('overlay1'),
	popup1 = document.getElementById('popup1'),
	btnCerrarPopup1 = document.getElementById('btn-cerrar-popup1');

btnAbrirPopup1.addEventListener('click', function(){
	overlay1.classList.add('active');
	popup1.classList.add('active');
});

btnCerrarPopup1.addEventListener('click', function(e){
	e.preventDefault();
	overlay1.classList.remove('active');
	popup1.classList.remove('active');
});


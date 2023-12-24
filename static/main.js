var $request;

function setProp(a,b) {
    return document.documentElement.style.setProperty(a,b)
}

function getProp(a) {
    return document.documentElement.style.getPropertyValue(a)
}

// Настройка цветов
setProp('--main-blue', '#dd944c');
setProp('--blue-hover', 'rgba(221, 148, 76, 0.2)');
setProp('--blue-light', 'rgba(221, 148, 76, 0.05)');

setProp('--main-red', '#dd4c4c');
setProp('--red-hover', 'rgba(221, 76, 76, 0.2)');
setProp('--red-light', 'rgba(221, 76, 76, 0.05)');

let blue_main  = getProp('--main-blue');
let blue_hover = getProp('--blue-hover');
let blue_light = getProp('--blue-light');

let red_main  = getProp('--main-red');
let red_hover = getProp('--red-hover');
let red_light = getProp('--red-light');

//setProp('--caption-url', 'url(' + file_name + ')');

function redSetup() {
    setProp('--main-blue',  red_main);
    setProp('--blue-hover', red_hover);
    setProp('--blue-light', red_light);
}

function blueSetup() {
    setProp('--main-blue',  blue_main);
    setProp('--blue-hover', blue_hover);
    setProp('--blue-light', blue_light);
}

$(document).ready(function() {
    let dropzone = $('#dropzone');
    let _done = false
    let dots = ''

    document.getElementById('myForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevents the default form submission action
    
        var fl = document.getElementById('check_flares');
        var sh = document.getElementById('check_shadows');
        var isFlares = fl.checked ? 1 : 0;
        var isShadows = sh.checked ? 1 : 0;

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/process_data', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({ flares: isFlares, shadows: isShadows }));
    });

    dropzone.on('dragover', function(event) {
        event.preventDefault();
        dropzone.addClass('hover');
    });

    dropzone.on('dragleave', function() {
        dropzone.removeClass('hover');
    });

    dropzone.on('drop', function(event) {
        event.preventDefault();
        dropzone.removeClass('hover');
        handleDroppedFiles(event.originalEvent.dataTransfer.files);
    });
    
    dropzone.on('click', function() {
        $('#fileInput').click();
    });
    
    function clearBlock() {
        return new Promise(() => {
            setTimeout(() => {
                $('#result-container').hide()
            }, 500);
        });
    }
    
    $('#fileInput').on('change', function() {        
        clearBlock()
        // Прерывание AJAX запроса (если все еще активен)
        if ($request != null){ 
            $request.abort();
            _done = false
            $request = null;
        }
        // Обработка файлов
        setProp('--caption-url', 'url()');
        handleDroppedFiles(this.files);
    });
    
    // Анимация загрузки ответа
    const countdownEl = document.querySelector("#uploadh2");
    function countdown(exception) {
        var int = setInterval(function() {
            if (!_done || exception) {
                if ((dots += '.').length == 4) 
                    dots = '';
                countdownEl.textContent = 'Processing your request ' + dots + '\xa0'.repeat(3-dots.length)
            }
            else {
                dots = ''
                clearInterval(int)
            }
        }, 500);
    }
    
    // Изменение цвета в случае ошибки
    function applyColors(exception, error_type) {
        if (exception) {
            redSetup()
            countdownEl.textContent = error_type
        }
        else {
            blueSetup()
            countdown(exception)
        }
    }

    function handleDroppedFiles(files) {
        const maxFileSize = 5242880;
        const imgTypes = ['image/jpg', 'image/jpeg', "image/png", "image/webp"]
        
        let file = files[0]
        let err = false
        let err_type = ''
        
        if (file.size > maxFileSize) {
            err = true
            err_type = 'File size is too large.'
        }
        
        if (!imgTypes.includes(file.type)) {
            err = true
            err_type = 'File format is invalid.'
        }
        
        applyColors(err, err_type)

        _done = false
        dots = ''
        
        let formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            if (!err) {
                formData.append('image', files[i]);
            }
        }
        
        $request = $.ajax({
            url: '/',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                countdownEl.textContent = 'Select an image for further captioning';
                $('#result-container').show();
                let input_file = data.input
                let output_file = data.result

                $('#img-input').html('<a href=""' + input_file + '"">\
                <img src="' + input_file + '"></a>')
                $('#img-result').html('<a href=""' + output_file + '"">\
                <img src="' + output_file + '"></a>')
                
                _done = true
            },
            error: function(errorMsg) {
                console.log('Error uploading file:', errorMsg);
            }
        });
    }
});

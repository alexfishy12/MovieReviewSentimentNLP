jQuery(function() {
    console.log("Document ready.")
    $("#response").hide()
    $("#error").hide()
    event_handler()
})

function event_handler(){
    $("form#input_review").on('submit', get_prediction)

    $("button#retry").on('click', function(){
        window.location.reload()
    })
}


function get_prediction(e){
    e.preventDefault()
    console.log("Receiving review...")
    var fd = new FormData();
    var review_text = $("textarea#review").val()
    fd.append('review_text', review_text)

    console.log(review_text)

    calculate_sentiment(fd).then(function(response){
        console.log(response)
        if (response.response.predictions != undefined) {
            $("#error").hide()

            var prediction_label = response.response.predictions
            var prediction_text = "undefined"
            if (prediction_label == 0) {
                $("span#prediction").css("color", "red")
                prediction_text = "negative"
            }
            else if (prediction_label == 1) {
                $("span#prediction").css("color", "green");
                prediction_text = "positive"
            }

            var probabilities = response.response.probabilities
            $("span#negative").html(probabilities.negative)
            $("span#positive").html(probabilities.positive)

            $("span#prediction").html(prediction_text)
            $("#response").show()
        }
        else {
            $("#response").hide()
            $("#error").html("Error loading a prediction.")
            $("#error").show()
        }
    })
}

function calculate_sentiment(fd) {
    console.log("Calculating sentiment...")
    return new Promise(function(resolve) {
        $.ajax({
            url: 'predict',
            type: 'POST',
            dataType: "json",
            processData: false,
            contentType: false,
            data: fd,
            success: function (response, status) {
                console.log('AJAX Success.');
                resolve(response);
            },
            error: function (XMLHttpRequest, textStatus, errorThrown) {
                console.log('AJAX Error:' + textStatus);
                resolve("Error " . textStatus);
                $("#response").hide()
                $("#error").html("Error loading a prediction.")
                $("#error").show()
            }
        })
    });
}
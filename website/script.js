jQuery(function() {
    console.log("Document ready.")
    event_handler()
})

function event_handler(){
    $("form#input_review").on('submit', get_prediction)
}


function get_prediction(e){
    e.preventDefault()
    console.log("Receiving review...")
    var fd = new FormData();
    var review_text = $("input#review").val()
    fd.append('review_text', review_text)

    console.log(review_text)

    calculate_sentiment(fd).then(function(response){
        console.log(response)
    })
}

function calculate_sentiment(fd) {
    console.log("Calculating sentiment...")
    return new Promise(function(resolve) {
        $.ajax({
            url: '../cgi-bin/sentiment_analysis/RunFineTunedModel.py',
            type: 'POST',
            dataType: "text",
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
            }
        })
    });
}
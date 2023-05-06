jQuery(function() {
    console.log("Document ready.")
    $("form#input_review").on('submit', get_prediction)
})

function get_prediction(e){
    e.preventDefault()
    console.log("Receiving review...")
    var fd = new FormData();
    var review_text = $("input#review").val()
    fd.append('review_text', review_text)

    console.log(review_text)

    new Promise(function(resolve) {
        $.ajax({
            url: 'RunFineTunedModel.py',
            dataType: 'text',
            type: 'POST',
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
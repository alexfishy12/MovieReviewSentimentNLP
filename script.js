jQuery(function() {
    console.log("Document ready.")
    $("form#input_review").on('submit', get_prediction)
})

function get_prediction(e){
    e.preventDefault()
    console.log("Receiving review...")
    var review = $("input#review").val()
    console.log(review)
}
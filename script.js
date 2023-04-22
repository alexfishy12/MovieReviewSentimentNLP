function respond() {
    event.preventDefault();
    document.getElementById("response").style.display="block";
    console.log("Responded!");
}

function reloadPage() {
    location.reload();
}
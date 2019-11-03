function changeImage() {
    var image = document.getElementById('myImage');
    if (image.src.match("img0")) {
        //Photo by Jametlene Reskp on Unsplash
        image.src = "../../static/jpg/img1.jpg";
        //Photo by ipet photo on Unsplash
        return;
    }
    if (image.src.match("img1")) {
        image.src = "../../static/jpg/img2.jpg";
        //Photo by samuel sng on Unsplash
        return;
    }
    if (image.src.match("img2")) {
        image.src = "../../static/jpg/img3.jpg";
        //Photo by Jametlene Reskp on Unsplash
        return;
    }
    if (image.src.match("img3")) {
        image.src = "../../static/jpg/img4.jpg";
        //Photo by Arjan Stalpers on Unsplash
        return;
    }
    if (image.src.match("img4")) {
        image.src = "../../static/jpg/img5.jpg";
        //Photo by Edson Torres on Unsplash
        return;
    }
    if (image.src.match("img5")) {
        image.src = "../../static/jpg/img6.jpg";
        //Photo by Julian Dutton on Unsplash
        return;
    }
    if (image.src.match("img6")) {
        image.src = "../../static/jpg/img7.jpg";
        //Photo by Andrii Podilnyk on Unsplash
        return;
    }
    if (image.src.match("img7")) {
        image.src = "../../static/jpg/img8.jpg";
        //Photo by Erin Wilson on Unsplash
        return;
    }
    if (image.src.match("img8")) {
        image.src = "../../static/jpg/img9.jpg";
        //Photo by Berkay Gumustekin on Unsplash
        return;
    }
    if (image.src.match("img9")) {
        image.src = "../../static/jpg/img10.jpg";
        //Photo by Lydia Tan on Unsplash
        return;
    }
    if (image.src.match("img10")) {
        image.src = "../../static/jpg/img0.jpg";
        return;
    }
    else {
        image.src = "../../static/jpg/img0.jpg";
        return;
    }
}

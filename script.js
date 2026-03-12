const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";
});
async function predictImage() {
    const file = imageInput.files[0];
    if (!file) {
        alert("Please upload an image first");
        return;
    }
    const formData = new FormData();
    formData.append("image", file);
    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        document.getElementById("resultCard").classList.remove("hidden");
        document.getElementById("resultLabel").innerText = data.prediction;
        document.getElementById("confidenceValue").innerText =
            data.confidence + "%";
    } catch (err) {
        alert("Server error. Please try again.");
    }
}

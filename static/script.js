// function startListening() {
//   fetch("/listen")
//     .then(res => res.json())
//     .then(data => {
//       const output = document.getElementById("output");
//       const spokenText = document.getElementById("spokenText");

//       output.innerHTML = "";

//       if (data.type === "error") {
//         spokenText.innerText = data.message;
//         return;
//       }

//       spokenText.innerText = "You said: " + data.text;

//       // Create one single image (video frame)
//       const videoFrame = document.createElement("img");
//       videoFrame.width = 300;
//       videoFrame.style.borderRadius = "12px";
//       videoFrame.style.boxShadow = "0 0 15px purple";
//       output.appendChild(videoFrame);

//       // If it is a GIF, just play it like a video
//       if (data.type === "gif") {
//         videoFrame.src = data.path;
//         return;
//       }

//       // If it is letters, play them like a video sequence
//       if (data.type === "letters") {
//         let index = 0;

//         function playVideoLikeSequence() {
//           if (index >= data.images.length) return;
//           videoFrame.src = data.images[index];
//           index++;
//           setTimeout(playVideoLikeSequence, 500);
//         }

//         playVideoLikeSequence();
//       }
//     });
// }

// function startCamera() {
//   const output = document.getElementById("output");
//   const cam = document.getElementById("cameraFeed");

//   output.innerHTML = "";
//   cam.style.display = "block";
//   cam.src = "/video_feed";
// }

// function textToSign() {
//   const text = document.getElementById("textInput").value;

//   if (text.trim() === "") {
//     alert("Please enter some text first");
//     return;
//   }

//   fetch("/text_to_sign", {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json"
//     },
//     body: JSON.stringify({ text: text })
//   })
//     .then(res => res.json())
//     .then(data => {
//       const output = document.getElementById("output");
//       const spokenText = document.getElementById("spokenText");

//       output.innerHTML = "";
//       spokenText.innerText = "Text: " + data.text;

//       // Video-like animation (one frame at a time)
//       const videoFrame = document.createElement("img");
//       videoFrame.width = 300;
//       videoFrame.style.borderRadius = "12px";
//       videoFrame.style.boxShadow = "0 0 15px purple";
//       output.appendChild(videoFrame);

//       let index = 0;

//       function playSequence() {
//         if (index >= data.images.length) return;
//         videoFrame.src = data.images[index];
//         index++;
//         setTimeout(playSequence, 500);
//       }

//       playSequence();
//     });
// }

// function startCamera() {
//   const cam = document.getElementById("cameraFeed");
//   cam.src = "/video_feed";
//   cam.style.display = "block";
// }

// document.addEventListener("visibilitychange", () => {
//   const cam = document.getElementById("cameraFeed");

//   if (document.hidden) {
//     cam.src = "";
//   } else {
//     cam.src = "/video_feed";
//   }
// });


function startListening() {
    const output = document.getElementById("output");
    const spokenText = document.getElementById("spokenText");

    // ✅ Show listening status immediately
    spokenText.innerText = "Listening...";
    output.innerHTML = "";

    fetch("/listen")
    .then(res => res.json())
    .then(data => {

        output.innerHTML = "";

        if (data.type === "error") {
            spokenText.innerText = data.message;
            return;
        }

        spokenText.innerText = "You said: " + data.text;

        // 🎥 Single image acting like a video frame
        const videoFrame = document.createElement("img");
        videoFrame.width = 300;
        videoFrame.style.borderRadius = "12px";
        videoFrame.style.boxShadow = "0 0 15px purple";

        output.appendChild(videoFrame);

        if (data.type === "gif") {
            videoFrame.src = data.path;
            return;
        }

        if (data.type === "letters" || data.type === "sequence") {
    let index = 0;

    function playVideoLikeSequence() {
        if (index >= data.images.length) return;

        videoFrame.src = data.images[index];
        index++;
        setTimeout(playVideoLikeSequence, 500);
    }

    playVideoLikeSequence();
}
    })
    .catch(() => {
        spokenText.innerText = "Error listening to microphone";
    });
}

// function startListening() {
//     fetch("/listen")
//     .then(res => res.json())
//     .then(data => {
//         const output = document.getElementById("output");
//         const spokenText = document.getElementById("spokenText");

//         output.innerHTML = "";

//         if (data.type === "error") {
//             spokenText.innerText = data.message;
//             return;
//         }

//         spokenText.innerText = "You said: " + data.text;

//         // ✅ Create ONE single image (video frame)
//         const videoFrame = document.createElement("img");
//         videoFrame.width = 300;
//         videoFrame.style.borderRadius = "12px";
//         videoFrame.style.boxShadow = "0 0 15px purple";

//         output.appendChild(videoFrame);

//         // ✅ If it is a GIF, just play it like a video
//         if (data.type === "gif") {
//             videoFrame.src = data.path;
//             return;
//         }

//         if (data.type === "letters" || data.type === "sequence") {
//     let index = 0;

//     function playVideoLikeSequence() {
//         if (index >= data.images.length) return;

//         videoFrame.src = data.images[index];
//         index++;
//         setTimeout(playVideoLikeSequence, 500);
//     }

//     playVideoLikeSequence();
// }

//         // // ✅ If it is letters, play them like a VIDEO SEQUENCE
//         // if (data.type === "letters") {
//         //     let index = 0;

//         //     function playVideoLikeSequence() {
//         //         if (index >= data.images.length) return;

//         //         videoFrame.src = data.images[index];
//         //         index++;

//         //         // 🎥 Frame rate control (milliseconds)
//         //         setTimeout(playVideoLikeSequence, 500);
//         //     }

//         //     playVideoLikeSequence();
//         // }
//     });
// }
function startCamera() {
    const output = document.getElementById("output");
    const cam = document.getElementById("cameraFeed");

    output.innerHTML = "";
    cam.style.display = "block";
    cam.src = "/video_feed";
}
function textToSign() {
    const text = document.getElementById("textInput").value;
    const output = document.getElementById("output");
    const spokenText = document.getElementById("spokenText");

    if (text.trim() === "") {
        alert("Please enter some text first");
        return;
    }

    spokenText.innerText = "Processing...";
    output.innerHTML = "";

    fetch("/text_to_sign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        output.innerHTML = "";
        spokenText.innerText = "Text: " + data.text;

        const videoFrame = document.createElement("img");
        videoFrame.width = 300;
        videoFrame.style.borderRadius = "12px";
        videoFrame.style.boxShadow = "0 0 15px purple";
        output.appendChild(videoFrame);

        // ✅ CASE 1: Phrase-level GIF
        if (data.type === "gif") {
            videoFrame.src = data.path;
            return;
        }

        // ✅ CASE 2: Sequence (letters / word gifs)
        if (data.type === "sequence") {
            let index = 0;

            function playSequence() {
                if (index >= data.images.length) return;
                videoFrame.src = data.images[index];
                index++;
                setTimeout(playSequence, 500);
            }

            playSequence();
        }
    })
    .catch(() => {
        spokenText.innerText = "Error processing text";
    });
}

function startCamera() {
    const cam = document.getElementById("cameraFeed");
    cam.src = "/video_feed";
    cam.style.display = "block";
}
document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
        document.getElementById("cameraFeed").src = "";
    } else {
        document.getElementById("cameraFeed").src = "/video_feed";
    }
});

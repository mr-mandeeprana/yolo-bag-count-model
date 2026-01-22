# How to Connect Your Camera to the Model

Follow these steps to link your IP camera (like Hikvision) to the bag counting code.

## Step 1: Physical Connection
1.  **Network:** Connect your camera to your network router or a PoE switch.
2.  **IP Address:** Ensure your computer and the camera are on the same network (e.g., both start with `192.168.1.X`).
3.  **Static IP:** It is highly recommended to set a "Static IP" in your camera settings so it doesn't change when the power goes out.

---

## Step 2: Camera Configuration (Web Interface)
1.  Open your browser and type the camera's IP (e.g., `http://192.168.1.100`).
2.  Log in with your admin username and password.
3.  Go to **Configuration** → **Network** → **Advanced Settings** → **RTSP**.
4.  Ensure RTSP is **Enabled** (usually on Port 554).
5.  (Optional but Recommended) Go to **Video/Audio** and set the resolution to **1080p** and 25-30 FPS.

---

## Step 3: Get the RTSP URL
The "Source" for the code is not just the IP. It is a specific URL. For Hikvision, use this format:

`rtsp://admin:YOUR_PASSWORD@192.168.1.100:554/Streaming/Channels/101`

*   `admin`: Your username
*   `YOUR_PASSWORD`: Your password
*   `192.168.1.100`: Your camera's IP
*   `101`: The main camera channel

---

## Step 4: Test the Connection
I have created a script called `test_camera.py` to help you verify the link without running the expensive AI model.

**Run this command:**
```powershell
python test_camera.py --source "rtsp://admin:YOUR_PASSWORD@YOUR_IP:554/Streaming/Channels/101"
```

If you see a window with live video, the connection is successful!

---

## Step 5: Start Counting
Once the test works, you can run the actual bag counting model:

```powershell
python src/inference_video.py --weights "models/weights/best.pt" --source "rtsp://admin:YOUR_PASSWORD@YOUR_IP:554/Streaming/Channels/101"
```

## Testing Without a Physical Camera

If you don't have the Hikvision camera with you yet, you can still test your code using these methods.

### Option A: Use Your Smartphone (Recommended)
You can turn your phone into an IP camera in 2 minutes:
1.  **Download:** Install the **"IP Webcam"** app (Android) or **"iVCam"** (iOS).
2.  **Start Server:** Open the app and click "Start Server."
3.  **Get the URL:** The app will show an IP (e.g., `192.168.1.5:8080`).
4.  **The Code Source:**
    `rtsp://192.168.1.5:8080/h264_pcm.sdp` (format varies by app).

### Option B: Use a Video File (Already Supported)
The test script I wrote for you (`test_camera.py`) can treat a video file as if it were a camera. This lets you check if the display and logic are working.

Run this command:
```powershell
python test_camera.py --source "WhatsApp Video 2026-01-20 at 12.38.18.mp4"
```

### Option C: Use a Public RTSP Test Stream
If your PC is connected to the internet, you can use a free online test stream:
```powershell
python test_camera.py --source "rtsp://rtspstream:260b9432f913165b6f00159cd0b42c6c@zephyr.rtsp.stream/pattern"
```
*(Note: Online streams can be laggy depending on your internet speed).*

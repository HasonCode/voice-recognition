# Deploy Hey Jetson listener on Jetson Orin Nano (JetPack 5)

This guide gets a **readily listening** Hey Jetson bot running on Jetson Orin Nano with JetPack 5 (L4T 35.x, Ubuntu 20.04). **JetPack 5 uses Python 3.8**; this project supports Python 3.8 and the steps below assume it. The service starts on boot and keeps the listener running.

---

## 1. Prerequisites on the Jetson

- **JetPack 5** (e.g. 5.0.2 / 5.1.x) with L4T 35.x and Ubuntu 20.04.
- **Python 3.8**: System Python on JetPack 5 is 3.8. Use it for the project (no need to install another Python).
- **Microphone**: USB or onboard mic; note the device index (see step 4).
- **Network** (optional): For edge-tts (AI voice); otherwise pyttsx3 works offline.

---

## 2. Copy the project onto the Jetson

From your dev machine (or clone on the Jetson):

```bash
# On Jetson (or SCP/rsync from host)
cd ~
git clone <your-repo-url> voice_recognition
cd voice_recognition
```

Or copy the project folder (e.g. with `scp -r voice_recognition jetson@<ip>:~/`).

---

## 3. Python 3.8 environment and dependencies

JetPack 5 ships with Python 3.8. Create a venv with it so the system Python stays unchanged.

```bash
cd ~/voice_recognition
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install dependencies. The project’s `pyproject.toml` allows Python 3.8 and uses 3.8‑compatible version bounds (e.g. `numpy>=1.20,<3.0`). On the Jetson you can either install from the project or use the 3.8-specific requirements file:

**Option A – Install from project (recommended if all deps have 3.8 wheels)**

```bash
pip install -e .
```

**Option B – Install core deps from requirements, then NeMo separately**

```bash
pip install -r docs/requirements-jetson38.txt
# Then install PyTorch and NeMo per NVIDIA Jetson docs (JetPack/CUDA-specific):
# https://docs.nvidia.com/nemo-framework/user-guide/latest/jetson.html
# e.g. pip install torch nemo_toolkit[asr]
```

**TTS (for voice readback)**

```bash
pip install pyttsx3
sudo apt install -y espeak   # offline TTS backend on Linux
# Optional, for AI-style voice (needs internet when generating):
# pip install edge-tts
```

NeMo / nemo_toolkit on Jetson often require a specific PyTorch build and sometimes a Jetson wheel; follow [NVIDIA NeMo Jetson documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/jetson.html).

---

## 4. Audio: default microphone

List capture devices:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Note the **input** device index. If the default is wrong, use `--device <index>` when running the listener (or in the systemd service, see below).

```bash
# Test mic and listener (inside venv)
source .venv/bin/activate
python hey_jetson_listener.py --device 0
# Say "Hey Jetson" then something; pause. You should see [Response] and hear TTS.
```

---

## 5. Run as a systemd service (start on boot, always listening)

So the bot is **readily listening** after power-on:

1. **Run script**

Use the provided **`deploy/run_hey_jetson.sh`**: it uses the project root (directory above `deploy/`) and activates `.venv` or `venv` if present. Make it executable:

```bash
chmod +x deploy/run_hey_jetson.sh
```

To force a specific project path (e.g. for systemd), set `VOICE_RECOGNITION_ROOT` to the project root.

2. **Install the systemd unit** (copy the provided `deploy/hey-jetson.service` and edit user/path):

```bash
# Edit paths and user in deploy/hey-jetson.service (YOUR_JETSON_USER and project path), then:
sudo cp deploy/hey-jetson.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hey-jetson
sudo systemctl start hey-jetson
sudo systemctl status hey-jetson
```

3. **Useful commands**

- Logs: `journalctl -u hey-jetson -f`
- Restart: `sudo systemctl restart hey-jetson`
- Stop: `sudo systemctl stop hey-jetson`
- Disable at boot: `sudo systemctl disable hey-jetson`

---

## 6. Optional: TTS and Llama on the edge

- **TTS**: For offline, `pyttsx3` + `espeak` (`sudo apt install espeak`) is enough. For AI-style voice, `pip install edge-tts` (needs internet when generating).
- **Llama**: To answer “Hey Jetson” with an LLM and then speak the answer, use a GGUF model and `llama_transcript_edge_example.py` or wire Llama into the Hey Jetson `on_response` callback; see the main docs for Llama on the edge.

---

## 7. Summary

| Step | Action |
|------|--------|
| 1 | JetPack 5, mic (Python 3.8 is system default) |
| 2 | Clone/copy project to Jetson |
| 3 | Create venv with `python3.8 -m venv .venv`, install deps (and NeMo for Jetson) |
| 4 | Find mic device index and test `hey_jetson_listener.py` |
| 5 | Install `deploy/run_hey_jetson.sh` and `deploy/hey-jetson.service`, enable and start service |

After this, the Jetson Orin Nano will run the Hey Jetson listener at boot and keep it running until you stop or disable the service.

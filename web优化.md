好的，我们可以通过改进CSS样式来让界面看起来更现代化、更美观。我会采用一些常见的现代设计原则，比如：

*   **更清晰的排版和字体：** 使用现代的无衬线字体。
*   **更好的颜色方案：** 使用更和谐、专业的颜色。
*   **卡片式设计 (Cards)：** 将各个功能区（如控制、统计、日志）用卡片包裹起来，增加视觉区隔和深度。
*   **间距和留白：** 合理使用 `padding` 和 `margin` 来让元素呼吸。
*   **圆角和阴影：** 适当使用 `border-radius` 和 `box-shadow` 增加柔和感和层次感。
*   **一致性：** 按钮、输入框等元素风格统一。

下面是修改后的 `templates/index.html` 文件，主要改动在 `<style>` 标签内，以及对HTML结构做了一些微调以配合新的样式（例如，将视频源输入框和其按钮组合）。

**`templates/index.html` (更新后的版本)**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Worker Safety Detection System</title>
    <style>
        /* Basic Reset & Modern Font */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            background-color: #f0f2f5; /* Light gray background */
            color: #333;
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        header {
            background-color: #1f2937; /* Darker, modern header */
            color: white;
            padding: 1.25em 1em;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        header h1 {
            margin: 0;
            font-size: 1.75em;
            font-weight: 600;
        }

        .container {
            display: flex;
            flex: 1;
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
            padding: 20px;
            gap: 20px; /* Spacing between main columns */
            max-width: 1800px; /* Max width for very large screens */
            margin: 20px auto; /* Center the content */
        }

        .video-container {
            flex: 3; /* Give more space to video */
            min-width: 640px;
            background-color: #000;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden; /* Ensures image corners are clipped if image is larger */
            display: flex; /* To center the image if it's smaller */
            align-items: center;
            justify-content: center;
        }
        .video-container img {
            display: block;
            width: 100%;
            height: auto;
            max-height: 80vh; /* Prevent video from being too tall */
        }

        .controls-stats-container {
            flex: 1;
            min-width: 340px; /* Slightly wider for better spacing */
            display: flex;
            flex-direction: column;
            gap: 20px; /* Spacing between panels */
        }

        .panel {
            background-color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Softer shadow */
        }
        .panel h2 {
            margin-top: 0;
            font-size: 1.25em;
            color: #111827; /* Darker text for headings */
            border-bottom: 1px solid #e5e7eb; /* Lighter border */
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-weight: 600;
        }

        /* Input Group for Source URL */
        .input-group {
            display: flex;
            margin-bottom: 15px;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 10px 14px;
            border: 1px solid #d1d5db; /* Tailwind-ish gray */
            border-radius: 6px 0 0 6px; /* Rounded left corners */
            font-size: 0.95em;
            transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #3b82f6; /* Blue focus */
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
        }
        .input-group button {
            border-radius: 0 6px 6px 0 !important; /* Important to override general button styles */
            margin: 0 !important; /* Remove default button margin */
            padding: 10px 18px !important;
        }


        button {
            padding: 10px 20px;
            margin: 8px 0; /* Vertical margin */
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 500;
            transition: background-color 0.2s ease, transform 0.1s ease;
            width: 100%; /* Make buttons in control panel full-width */
            box-sizing: border-box;
        }
        button:hover {
            filter: brightness(90%);
        }
        button:active {
            transform: translateY(1px);
        }

        #startBtn { background-color: #10b981; color: white; } /* Emerald Green */
        #stopBtn { background-color: #ef4444; color: white; } /* Red */
        #updateSourceBtn { background-color: #3b82f6; color: white; } /* Blue */


        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background-color: #f9fafb; /* Very light gray for stat items */
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #e5e7eb;
        }
        .stat-item span { /* The Label: People, Fallen etc. */
            display: block;
            font-size: 0.85em;
            color: #6b7280; /* Medium gray for label */
            margin-bottom: 5px;
        }
        .stat-item strong { /* The Number */
            display: block;
            font-size: 2.2em;
            font-weight: 700;
            color: #1f2937; /* Dark gray for number */
        }
        /* Specific colors for stat numbers */
        #fallenCount { color: #ef4444; }
        #stationaryCount { color: #f59e0b; } /* Amber/Orange */
        #noHelmetCount { color: #8b5cf6; } /* Violet */


        #alertsLog {
            list-style-type: none;
            padding: 0;
            height: 280px; /* Increased height */
            overflow-y: auto;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            font-size: 0.9em;
            background-color: #fdfdfd;
        }
        #alertsLog li {
            padding: 10px 15px;
            border-bottom: 1px solid #f3f4f6; /* Lighter separator */
            word-break: break-word;
        }
        #alertsLog li:last-child {
            border-bottom: none;
        }
        /* Optional: Zebra striping if desired, but might conflict with critical alert bg
        #alertsLog li:nth-child(odd) { background-color: #f9fafb; } */

        .alert-critical {
            color: #991b1b; /* Darker red for text */
            background-color: #fee2e2 !important; /* Light red background, !important to override potential zebra */
            font-weight: 500;
        }
        /* Example for other types of alerts if you add them */
        .alert-warning {
            color: #92400e; /* Darker amber */
            background-color: #fef3c7 !important; /* Light amber */
        }

        .system-status {
            padding: 12px 15px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.95em;
            border: 1px solid transparent;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
        /* JS will add these classes */
        .status-running { background-color: #d1fae5; color: #065f46; border-color: #a7f3d0;} /* Greenish */
        .status-stopped { background-color: #fee2e2; color: #991b1b; border-color: #fecaca;} /* Reddish */
        .status-unknown { background-color: #e5e7eb; color: #4b5563; border-color: #d1d5db;} /* Neutral gray */


        /* Responsive adjustments */
        @media (max-width: 1200px) { /* Adjust breakpoint as needed */
             .video-container {
                min-width: 500px;
            }
            .controls-stats-container {
                min-width: 300px;
            }
        }
        @media (max-width: 992px) { /* Stack columns on smaller screens */
            .container {
                flex-direction: column;
                padding: 15px;
                gap: 15px;
            }
            .video-container, .controls-stats-container {
                min-width: 100%; /* Take full width when stacked */
                max-width: 100%;
            }
            .video-container img {
                max-height: 60vh;
            }
        }
        @media (max-width: 576px) {
            header h1 { font-size: 1.4em; }
            .panel { padding: 15px; }
            .panel h2 { font-size: 1.1em; margin-bottom: 15px;}
            button, input[type="text"] { font-size: 0.9em; }
            .stat-item strong { font-size: 1.8em; }
            #alertsLog { height: 200px; }
        }

    </style>
</head>
<body>
    <header>
        <h1>Worker Safety Detection System</h1>
    </header>

    <div class="container">
        <div class="video-container">
            <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>

        <div class="controls-stats-container">
            <div class="panel">
                <h2>System Control</h2>
                <div class="system-status status-unknown" id="systemStatus">Status: Initializing...</div> <!-- Initial class -->
                <div class="input-group"> <!-- Wrapped input and button -->
                    <input type="text" id="videoSource" value="{{ current_source }}" placeholder="Video Path, RTSP URL, or Webcam ID">
                    <button id="updateSourceBtn">Update</button>
                </div>
                <button id="startBtn">Start Detection</button>
                <button id="stopBtn">Stop Detection</button>
            </div>

            <div class="panel">
                <h2>Live Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-item"><span>People</span><strong id="peopleCount">0</strong></div>
                    <div class="stat-item"><span>Fallen</span><strong id="fallenCount" style="color: #ef4444;">0</strong></div>
                    <div class="stat-item"><span>Stationary</span><strong id="stationaryCount" style="color: #f59e0b;">0</strong></div>
                    <div class="stat-item"><span>No Helmet</span><strong id="noHelmetCount" style="color: #8b5cf6;">0</strong></div>
                </div>
            </div>

            <div class="panel">
                <h2>Event Log</h2>
                <ul id="alertsLog">
                    <li>Log will appear here...</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const updateSourceBtn = document.getElementById('updateSourceBtn');
        const videoSourceInput = document.getElementById('videoSource');
        const systemStatusDiv = document.getElementById('systemStatus');

        const peopleCountEl = document.getElementById('peopleCount');
        const fallenCountEl = document.getElementById('fallenCount');
        const stationaryCountEl = document.getElementById('stationaryCount');
        const noHelmetCountEl = document.getElementById('noHelmetCount');
        const alertsLogUl = document.getElementById('alertsLog');

        startBtn.addEventListener('click', () => {
            const source = videoSourceInput.value;
            fetch('/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ source: source })
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                // Status will be updated by the periodic fetch
            });
        });

        stopBtn.addEventListener('click', () => {
            fetch('/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
            });
        });

        updateSourceBtn.addEventListener('click', () => {
            const newSource = videoSourceInput.value;
            fetch('/update_source', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ source: newSource })
            })
            .then(response => response.json())
            .then(data => {
                // Using a more subtle way to show message, or just rely on status update
                if(data.success) {
                    console.info("Source update successful:", data.message);
                    // The status poller will eventually reflect the new source if detection is restarted
                } else {
                    console.error("Source update failed:", data.message);
                    alert("Error updating source: " + data.message); // Alert for errors
                }
            });
        });

        function updateStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                let statusText;
                let statusClass;

                if (data.running) {
                    statusText = 'Running';
                    statusClass = 'status-running';
                } else {
                    statusText = 'Stopped';
                    statusClass = 'status-stopped';
                }
                systemStatusDiv.textContent = `Status: ${statusText} | Source: ${data.source || 'N/A'}`;
                systemStatusDiv.className = `system-status ${statusClass}`; // Update class for styling

                peopleCountEl.textContent = data.stats.people_count;
                fallenCountEl.textContent = data.stats.fallen_count;
                stationaryCountEl.textContent = data.stats.stationary_count;
                noHelmetCountEl.textContent = data.stats.no_helmet_count;

                // Efficiently update alerts log
                const newAlerts = data.alerts.slice(0, 30); // Show latest 30
                // Check if update is needed (simple check, could be more robust)
                if (alertsLogUl.children.length !== newAlerts.length || 
                    (alertsLogUl.firstChild && newAlerts.length > 0 && alertsLogUl.firstChild.textContent !== escapeHtml(newAlerts[0]))) {
                    
                    alertsLogUl.innerHTML = ''; // Clear existing
                    if (newAlerts.length === 0) {
                        const li = document.createElement('li');
                        li.textContent = "No new events.";
                        alertsLogUl.appendChild(li);
                    } else {
                        newAlerts.forEach(alertMsg => {
                            const li = document.createElement('li');
                            li.textContent = escapeHtml(alertMsg);
                            if (alertMsg.includes("FALLEN") || alertMsg.includes("NO HELMET") || alertMsg.includes("[ERROR]")) {
                                li.classList.add('alert-critical');
                            } else if (alertMsg.includes("[WARNING]")) {
                                // li.classList.add('alert-warning'); // If you define this style
                            }
                            alertsLogUl.appendChild(li);
                        });
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
                systemStatusDiv.textContent = 'Status: Error fetching status';
                systemStatusDiv.className = 'system-status status-stopped'; // Show as stopped on error
            });
        }

        function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') return '';
            return unsafe
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }

        setInterval(updateStatus, 2000); // Update every 2 seconds
        updateStatus(); // Initial call
    </script>
</body>
</html>
```

**主要改动点：**

1.  **CSS Styles (`<style>` block):**
    *   **Body & Font:** Switched to a system font stack for a modern feel. Light gray background (`#f0f2f5`).
    *   **Header:** Darker background, larger padding, subtle shadow.
    *   **Container:** Flexbox layout with `gap` for spacing. Max-width and centered.
    *   **Video Container:** Black background, rounded corners, shadow. `max-height` to prevent it from being excessively tall.
    *   **Panels (`.panel`):** White background, rounded corners, softer shadows, better padding. Panel headings are styled for clarity.
    *   **Input Group (`.input-group`):** The "Video Source" input and "Update" button are now grouped together for a common UI pattern. Input fields have focus styles.
    *   **Buttons:** Standardized look with rounded corners, hover and active states. Specific colors for Start (green), Stop (red), Update (blue). Buttons in the control panel are full-width.
    *   **Statistics (`.stats-grid`, `.stat-item`):** Stat items have a light background, border, and better typography for numbers and labels. Colors for specific stats (Fallen, Stationary, No Helmet) are applied directly to the number.
    *   **Alerts Log (`#alertsLog`):** Increased height, softer border, slightly different background for list items. Critical alerts have a distinct background and text color.
    *   **System Status (`.system-status`):** Dynamically styled based on whether the system is running or stopped using CSS classes (`status-running`, `status-stopped`, `status-unknown`) applied by JavaScript.
    *   **Responsiveness (`@media` queries):** Added basic media queries to stack columns on smaller screens and adjust some sizings.

2.  **HTML Structure:**
    *   The `input type="text"` for video source and the `updateSourceBtn` are wrapped in a `div` with class `input-group` to style them together.
    *   The "People", "Fallen" etc. labels in the stats section are now `<span>` elements, and the numbers are `<strong>` for better semantic structure and styling control.
    *   Initial class `status-unknown` for the system status div.

3.  **JavaScript (`<script>` block):**
    *   The `updateStatus` function now also sets a class on `systemStatusDiv` (`status-running` or `status-stopped`) to apply the new dynamic styles.
    *   Improved the alert log update logic to be a bit more efficient (clears and re-adds only if content seems to have changed significantly, or just show "No new events").
    *   `escapeHtml` function now checks if `unsafe` is a string before trying to replace.

**如何使用：**

1.  将上述代码完全替换掉您现有的 `templates/index.html` 文件。
2.  确保您的 `app.py` 仍在运行。
3.  刷新浏览器中的 `http://127.0.0.1:5000` 页面。

您应该会看到一个外观更现代、布局更清晰的界面。您可以根据自己的喜好进一步调整颜色、字体和间距。

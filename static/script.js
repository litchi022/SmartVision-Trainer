document.addEventListener('DOMContentLoaded', () => {
    const statusDiv = document.getElementById('status');
    const addClassBtn = document.getElementById('add-class-btn');
    const datasetContainer = document.getElementById('dataset-container');
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const trainBtn = document.getElementById('train-btn');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');
    const video = document.getElementById('video');
    const predictionResultDiv = document.getElementById('prediction-result');
    const trainingProgressContainer = document.getElementById('training-progress-container');
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const trainingLog = document.getElementById('training-log');
    const confidenceBarsContainer = document.getElementById('confidence-bars-container');
    const serverDatasetsList = document.getElementById('server-datasets-list');
    const deleteAllDatasetsBtn = document.getElementById('delete-all-datasets-btn');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadProgressBar = document.getElementById('upload-progress-bar');
    const exportBtn = document.getElementById('export-btn');
    const importForm = document.getElementById('import-form');
    const importFileInput = document.getElementById('import-file-input');
    const importBtn = document.getElementById('import-btn');

    let capturedClasses = new Set();
    // Use a WeakMap to store file lists for each input, avoiding memory leaks
    const fileInputStorage = new WeakMap();
    let stream = null; // To hold the camera stream
    let predictionSocket = null; // To hold the prediction WebSocket

    function renderDatasetList(classNames = []) {
        serverDatasetsList.innerHTML = '';
        if (classNames.length === 0) {
            serverDatasetsList.innerHTML = '<li>无</li>';
            deleteAllDatasetsBtn.style.display = 'none';
            return;
        }

        classNames.forEach(name => {
            const li = document.createElement('li');
            li.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 5px; border-bottom: 1px solid #f0f0f0;';
            li.innerHTML = `
                <span>${name}</span>
                <button class="delete-class-btn" data-class-name="${name}" style="background: none; border: none; color: #dc3545; cursor: pointer;">×</button>
            `;
            serverDatasetsList.appendChild(li);
        });
        deleteAllDatasetsBtn.style.display = 'inline-block';
    }

    async function checkServerStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                updateStatus('无法获取服务器状态。');
                return;
            }
            const status = await response.json();
            trainBtn.disabled = !status.datasets_available;
            startCameraBtn.disabled = !status.model_available;
            renderDatasetList(status.class_names);
            
            // Handle post-import action
            const postImportAction = sessionStorage.getItem('postImportAction');
            if (postImportAction === 'focusStep3' && status.model_available) {
                sessionStorage.removeItem('postImportAction');
                const step3Section = startCameraBtn.closest('.section');
                step3Section.scrollIntoView({ behavior: 'smooth' });
                step3Section.style.transition = 'background-color 0.5s';
                step3Section.style.backgroundColor = '#e7f3ff';
                setTimeout(() => {
                    step3Section.style.backgroundColor = 'white';
                }, 1500);
            }

        } catch (error) {
            updateStatus(`检查状态时出错: ${error.message}`);
        }
    }

    function updateStatus(message) {
        statusDiv.textContent = message;
    }

    function toggleRemoveButtons() {
        const groups = document.querySelectorAll('.dataset-group');
        groups.forEach(group => {
            const button = group.querySelector('.remove-class-btn');
            if (button) {
                button.style.display = groups.length > 1 ? 'inline-block' : 'none';
            }
        });
    }

    function renderFilePreview(fileInput) {
        const previewContainer = fileInput.parentElement.querySelector('.file-preview-container');
        const fileList = fileInputStorage.get(fileInput) || [];
        previewContainer.innerHTML = '';
        fileList.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'file-preview-item';
            item.innerHTML = `
                <span>${file.name}</span>
                <button type="button" class="remove-file-btn" data-index="${index}">×</button>
            `;
            previewContainer.appendChild(item);
        });
    }

    datasetContainer.addEventListener('click', (e) => {
        // Handle removing a whole class group
        if (e.target.classList.contains('remove-class-btn')) {
            e.target.parentElement.remove();
            toggleRemoveButtons();
        }
        // Handle removing a single file from the preview
        if (e.target.classList.contains('remove-file-btn')) {
            const fileInput = e.target.closest('.dataset-group').querySelector('input[type="file"]');
            const indexToRemove = parseInt(e.target.dataset.index, 10);
            let fileList = fileInputStorage.get(fileInput) || [];
            fileList.splice(indexToRemove, 1);
            fileInputStorage.set(fileInput, fileList);
            renderFilePreview(fileInput);
        }
    });

    datasetContainer.addEventListener('change', (e) => {
        if (e.target.type === 'file') {
            const files = Array.from(e.target.files);
            fileInputStorage.set(e.target, files);
            renderFilePreview(e.target);
        }
    });

    addClassBtn.addEventListener('click', () => {
        const newGroup = document.createElement('div');
        newGroup.className = 'dataset-group';
        newGroup.innerHTML = `
            <label>分类名称:</label>
            <input type="text" name="class_name" required placeholder="例如: another_class">
            <label>图片文件:</label>
            <input type="file" name="files" multiple required>
            <div class="file-preview-container"></div>
            <button type="button" class="remove-class-btn">×</button>
        `;
        datasetContainer.appendChild(newGroup);
        toggleRemoveButtons();
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        updateStatus('正在上传文件...');
        uploadBtn.disabled = true;
        uploadProgressContainer.style.display = 'block';
        uploadProgressBar.value = 0;

        const formData = new FormData();
        const classGroups = document.querySelectorAll('.dataset-group');
        let totalFiles = 0;

        classGroups.forEach(group => {
            const className = group.querySelector('input[name="class_name"]').value;
            const fileInput = group.querySelector('input[name="files"]');
            const files = fileInputStorage.get(fileInput) || [];
            
            if (className && files.length > 0) {
                files.forEach(file => {
                    formData.append('files', file);
                    formData.append('class_names', className);
                });
                totalFiles += files.length;
            }
        });

        if (totalFiles === 0) {
            updateStatus('请至少为一个分类选择一些文件。');
            uploadBtn.disabled = false;
            return;
        }

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const percentComplete = (event.loaded / event.total) * 100;
                uploadProgressBar.value = percentComplete;
            }
        });

        xhr.addEventListener('load', () => {
            uploadBtn.disabled = false;
            uploadProgressContainer.style.display = 'none';
            if (xhr.status >= 200 && xhr.status < 300) {
                updateStatus('数据集上传成功！现在可以开始训练了。');
                checkServerStatus(); // Re-check status to enable train button
            } else {
                try {
                    const error = JSON.parse(xhr.responseText);
                    updateStatus(`上传失败: ${error.detail || '服务器错误'}`);
                } catch (e) {
                    updateStatus(`上传失败: ${xhr.statusText}`);
                }
            }
        });

        xhr.addEventListener('error', () => {
            uploadBtn.disabled = false;
            uploadProgressContainer.style.display = 'none';
            updateStatus('上传出错：无法连接到服务器。');
        });

        xhr.addEventListener('abort', () => {
            uploadBtn.disabled = false;
            uploadProgressContainer.style.display = 'none';
            updateStatus('上传已取消。');
        });

        xhr.open('POST', '/upload_datasets/', true);
        xhr.send(formData);
    });

    trainBtn.addEventListener('click', async () => {
        updateStatus('正在发送训练请求...');
        trainBtn.disabled = true;
        try {
            const response = await fetch('/train/', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                updateStatus('模型训练已在后台开始。请查看下方进度。');
                trainingProgressContainer.style.display = 'block';
                checkServerStatus(); // Re-check status, might enable camera button
                connectToTrainingProgress(data.session_id);
            } else {
                const error = await response.json();
                updateStatus(`训练请求失败: ${error.detail || '未知错误'}`);
                trainBtn.disabled = false;
            }
        } catch (error) {
            updateStatus(`训练请求出错: ${error.message}`);
            trainBtn.disabled = false;
        }
    });

    function connectToTrainingProgress(sessionId) {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/train_progress/${sessionId}`);

        socket.onmessage = event => {
            const data = JSON.parse(event.data);
            if(data.error) {
                updateStatus(`训练错误: ${data.error}`);
                return;
            }
            
            if (data.total_epochs > 0) {
                const progress = (data.current_epoch / data.total_epochs) * 100;
                trainingProgressBar.value = progress;
            }
            trainingLog.textContent = data.log;

            if (data.status === "Completed") {
                updateStatus("训练完成！现在可以使用最新的模型进行预测。");
                socket.close();
                checkServerStatus(); // Re-check status when training completes
            }
        };

        socket.onclose = () => {
            updateStatus('训练监控连接已关闭。');
            checkServerStatus(); // Re-check status when training completes
        };

        socket.onerror = error => {
            updateStatus(`训练监控连接错误: ${error.message}`);
            console.error('Training WebSocket Error:', error);
        };
    }

    startCameraBtn.addEventListener('click', () => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            updateStatus('正在启动摄像头...');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(s => { // 's' is the stream
                    stream = s;
                    video.srcObject = stream;
                    video.play();
                    updateStatus('摄像头已启动。开始实时预测...');
                    startCameraBtn.style.display = 'none';
                    stopCameraBtn.style.display = 'inline-block';
                    startPredictionWebSocket();
                })
                .catch(err => {
                    updateStatus(`摄像头错误: ${err.message}`);
                    console.error("Error accessing camera: ", err);
                });
        } else {
            updateStatus('您的浏览器不支持摄像头访问功能。');
        }
    });

    stopCameraBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
        }
        if (predictionSocket) {
            predictionSocket.close();
            predictionSocket = null;
        }
        stopCameraBtn.style.display = 'none';
        startCameraBtn.style.display = 'inline-block';
        predictionResultDiv.textContent = '';
        confidenceBarsContainer.innerHTML = '<h3>实时置信度</h3>';
        updateStatus('摄像头已关闭。');
    });

    function startPredictionWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        predictionSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/predict`);

        predictionSocket.onopen = () => {
            updateStatus('与服务器连接成功，正在等待数据...');
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 480;

            const intervalId = setInterval(() => {
                if (predictionSocket && predictionSocket.readyState === WebSocket.OPEN) {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(blob => {
                        if (blob && predictionSocket.readyState === WebSocket.OPEN) {
                            predictionSocket.send(blob);
                        }
                    }, 'image/jpeg');
                } else {
                    clearInterval(intervalId);
                }
            }, 200); // Send a frame every 200ms for more responsiveness
        };

        predictionSocket.onmessage = event => {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                predictionResultDiv.textContent = `错误: ${data.error}`;
                predictionSocket.close();
                return;
            }

            // First message contains class names to build the UI
            if (data.class_names) {
                confidenceBarsContainer.innerHTML = '<h3>实时置信度</h3>'; // Clear previous bars
                data.class_names.forEach(className => {
                    const barDiv = document.createElement('div');
                    barDiv.style.marginBottom = '8px';
                    barDiv.innerHTML = `
                        <label style="display: inline-block; width: 120px;">${className}</label>
                        <progress id="progress-${className}" value="0" max="100" style="width: calc(100% - 180px);"></progress>
                        <span id="confidence-${className}" style="display: inline-block; width: 40px; text-align: right;">0%</span>
                    `;
                    confidenceBarsContainer.appendChild(barDiv);
                });
                // Start sending frames *after* UI is built
                updateStatus('正在发送视频帧...');
            }

            // Subsequent messages contain confidences
            if (data.confidences) {
                let topClass = '';
                let topConfidence = 0;
                for (const [className, confidence] of Object.entries(data.confidences)) {
                    if (confidence > topConfidence) {
                        topConfidence = confidence;
                        topClass = className;
                    }
                    const percentage = Math.round(confidence * 100);
                    const progressBar = document.getElementById(`progress-${className}`);
                    const confidenceSpan = document.getElementById(`confidence-${className}`);
                    if (progressBar) progressBar.value = percentage;
                    if (confidenceSpan) confidenceSpan.textContent = `${percentage}%`;
                }
                predictionResultDiv.textContent = `预测: ${topClass} (${Math.round(topConfidence * 100)}%)`;
            }
        };

        predictionSocket.onclose = () => {
            updateStatus('与服务器断开连接。');
            predictionResultDiv.textContent = '';
        };

        predictionSocket.onerror = error => {
            updateStatus(`WebSocket 错误: ${error.message}`);
            console.error('WebSocket Error:', error);
        };
    }

    exportBtn.addEventListener('click', () => {
        updateStatus('正在准备导出文件...');
        window.location.href = '/export_project/';
        setTimeout(() => {
            updateStatus('准备就绪');
        }, 2000);
    });

    importFileInput.addEventListener('change', () => {
        importBtn.disabled = !importFileInput.files.length > 0;
    });

    importForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = importFileInput.files[0];
        if (!file) {
            updateStatus('请选择一个 .zip 文件进行导入。');
            return;
        }

        sessionStorage.setItem('postImportAction', 'focusStep3');
        const confirmed = confirm("警告：导入项目将覆盖服务器上所有现存的数据集和模型。这个操作无法撤销。您确定要继续吗？");
        if (!confirmed) {
            updateStatus('导入操作已取消。');
            sessionStorage.removeItem('postImportAction'); // Clean up if cancelled
            return;
        }

        updateStatus('正在上传并导入项目...');
        importBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/import_project/', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                updateStatus(data.message);
                setTimeout(() => window.location.reload(), 2000);
            } else {
                updateStatus(`导入失败: ${data.detail || '未知错误'}`);
            }
        } catch (error) {
            updateStatus(`导入出错: ${error.message}`);
        } finally {
            importBtn.disabled = false;
        }
    });

    serverDatasetsList.addEventListener('click', async (e) => {
        if (!e.target.classList.contains('delete-class-btn')) return;
        
        const className = e.target.dataset.className;
        if (!confirm(`您确定要删除 '${className}' 这个分类下的所有图片吗？`)) return;

        try {
            const response = await fetch(`/dataset/${encodeURIComponent(className)}`, { method: 'DELETE' });
            const result = await response.json();
            updateStatus(response.ok ? result.message : `删除失败: ${result.detail}`);
            await checkServerStatus();
        } catch (error) {
            updateStatus(`删除时出错: ${error.message}`);
        }
    });

    deleteAllDatasetsBtn.addEventListener('click', async () => {
        if (!confirm('警告：您确定要删除服务器上所有的数据集吗？此操作不可恢复。')) return;

        try {
            const response = await fetch('/datasets/all', { method: 'DELETE' });
            const result = await response.json();
            updateStatus(response.ok ? result.message : `删除失败: ${result.detail}`);
            await checkServerStatus();
        } catch (error) {
            updateStatus(`清空数据集时出错: ${error.message}`);
        }
    });

    // Initial setup
    toggleRemoveButtons();
    checkServerStatus();
}); 
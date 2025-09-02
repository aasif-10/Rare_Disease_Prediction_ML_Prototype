// Theme Management
class ThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.applyTheme();
        this.setupEventListeners();
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);
        
        // Update theme toggle button
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            const lightIcon = themeToggle.querySelector('.light-icon');
            const darkIcon = themeToggle.querySelector('.dark-icon');
            
            if (this.currentTheme === 'dark') {
                lightIcon.style.display = 'none';
                darkIcon.style.display = 'block';
            } else {
                lightIcon.style.display = 'block';
                darkIcon.style.display = 'none';
            }
        }
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        
        // Add theme transition effect
        document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }

    setupEventListeners() {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
    }
}

// UI Manager for enhanced interactions
class UIManager {
    constructor() {
        this.setupEventListeners();
        this.setupAnimations();
    }

    setupEventListeners() {
        // Help modal
        const helpBtn = document.getElementById('helpBtn');
        const helpModal = document.getElementById('helpModal');
        const closeHelpModal = document.getElementById('closeHelpModal');

        if (helpBtn && helpModal) {
            helpBtn.addEventListener('click', () => this.showModal(helpModal));
        }

        if (closeHelpModal && helpModal) {
            closeHelpModal.addEventListener('click', () => this.hideModal(helpModal));
        }

        // Close modal on outside click
        if (helpModal) {
            helpModal.addEventListener('click', (e) => {
                if (e.target === helpModal) {
                    this.hideModal(helpModal);
                }
            });
        }

        // Character counter
        const textarea = document.getElementById('patientDescription');
        const charCount = document.getElementById('charCount');
        if (textarea && charCount) {
            textarea.addEventListener('input', () => {
                const count = textarea.value.length;
                charCount.textContent = count;
                
                // Add visual feedback
                if (count > 500) {
                    charCount.style.color = 'var(--warning-color)';
                } else if (count > 200) {
                    charCount.style.color = 'var(--accent-color)';
                } else {
                    charCount.style.color = 'var(--text-muted)';
                }
            });
        }

        // Enhanced form interactions
        this.setupFormEnhancements();
    }

    setupFormEnhancements() {
        const form = document.getElementById('analysisForm');
        if (form) {
            // Add focus effects to form elements
            const formElements = form.querySelectorAll('input, textarea, select');
            formElements.forEach(element => {
                element.addEventListener('focus', () => {
                    element.parentElement.classList.add('focused');
                });
                
                element.addEventListener('blur', () => {
                    element.parentElement.classList.remove('focused');
                });
            });
        }
    }

    setupAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        const animateElements = document.querySelectorAll('.info-card, .input-section, .info-section');
        animateElements.forEach(el => {
            observer.observe(el);
        });
    }

    showModal(modal) {
        modal.classList.add('show');
        document.body.style.overflow = 'hidden';
        
        // Add entrance animation
        modal.style.animation = 'slideInUp 0.3s ease';
    }

    hideModal(modal) {
        modal.classList.remove('show');
        document.body.style.overflow = '';
        
        // Add exit animation
        modal.style.animation = 'slideOutDown 0.3s ease';
        setTimeout(() => {
            modal.style.animation = '';
        }, 300);
    }

    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// File Upload Manager with drag and drop
class FileUploadManager {
    constructor() {
        this.currentFile = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        const fileInput = document.getElementById('imageFile');
        const fileUpload = document.getElementById('fileUpload');
        const previewImage = document.getElementById('previewImage');
        const removeFile = document.getElementById('removeFile');
        const fileInfo = document.getElementById('fileInfo');

        if (fileUpload) {
            fileUpload.addEventListener('click', () => fileInput.click());
            
            // Drag and drop events
            fileUpload.addEventListener('dragover', (e) => this.handleDragOver(e));
            fileUpload.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            fileUpload.addEventListener('drop', (e) => this.handleDrop(e));
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        if (removeFile) {
            removeFile.addEventListener('click', () => this.removeFile());
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.processFile(e.target.files[0]);
        }
    }

    processFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            uiManager.showNotification('Please select an image file', 'error');
            return;
        }

        // Validate file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            uiManager.showNotification('File size must be less than 16MB', 'error');
            return;
        }

        this.currentFile = file;
        this.displayFilePreview(file);
        this.updateFileInfo(file);
        
        // Add success notification
        uiManager.showNotification(`File "${file.name}" uploaded successfully`, 'success');
        
        // Add debug info
        addDebugInfo(`File selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
    }

    displayFilePreview(file) {
        const previewImage = document.getElementById('previewImage');
        const fileUpload = document.getElementById('fileUpload');
        
        if (previewImage && fileUpload) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                
                // Hide upload content
                const uploadContent = fileUpload.querySelector('.file-upload-content');
                if (uploadContent) {
                    uploadContent.style.display = 'none';
                }
            };
            reader.readAsDataURL(file);
        }
    }

    updateFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = fileInfo.querySelector('.file-name');
        
        if (fileInfo && fileName) {
            fileName.textContent = file.name;
            fileInfo.style.display = 'flex';
        }
    }

    removeFile() {
        this.currentFile = null;
        
        // Reset UI
        const previewImage = document.getElementById('previewImage');
        const fileUpload = document.getElementById('fileUpload');
        const fileInfo = document.getElementById('fileInfo');
        const uploadContent = fileUpload.querySelector('.file-upload-content');
        
        if (previewImage) previewImage.style.display = 'none';
        if (fileInfo) fileInfo.style.display = 'none';
        if (uploadContent) uploadContent.style.display = 'flex';
        
        // Reset file input
        const fileInput = document.getElementById('imageFile');
        if (fileInput) fileInput.value = '';
        
        addDebugInfo('File removed');
    }

    getCurrentFile() {
        return this.currentFile;
    }
}

// Patient Submission Manager
class PatientSubmissionManager {
    constructor() {
        this.API_BASE = 'http://localhost:5000/api';
        this.setupEventListeners();
    }

    setupEventListeners() {
        const form = document.getElementById('analysisForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitCase();
            });
        }
    }

    async submitCase() {
        const patientDescription = document.getElementById('patientDescription').value.trim();
        const currentFile = fileUploadManager.getCurrentFile();

        if (!patientDescription && !currentFile) {
            uiManager.showNotification('Please provide either a symptom description or upload an image.', 'error');
            return;
        }

        this.showLoading();
        this.hideMessages();

        const formData = new FormData();
        formData.append('patient_description', patientDescription);
        if (currentFile) {
            formData.append('image', currentFile);
        }

        try {
            const response = await fetch(`${this.API_BASE}/patient/submit`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.success) {
                this.showSuccess(data);
                uiManager.showNotification('Case submitted successfully!', 'success');
            } else {
                throw new Error(data.error || 'Case submission failed');
            }

        } catch (error) {
            addDebugInfo(`Request failed: ${error.message}`);
            uiManager.showNotification(`Case submission failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        const loading = document.getElementById('loading');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (loading) loading.style.display = 'block';
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.querySelector('.btn-text').style.display = 'none';
            analyzeBtn.querySelector('.btn-loading').style.display = 'inline-flex';
        }

        // Animate loading steps
        this.animateLoadingSteps();
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (loading) loading.style.display = 'none';
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('.btn-text').style.display = 'inline';
            analyzeBtn.querySelector('.btn-loading').style.display = 'none';
        }
    }

    animateLoadingSteps() {
        const steps = document.querySelectorAll('.loading-steps .step');
        let currentStep = 0;

        const interval = setInterval(() => {
            if (currentStep < steps.length) {
                steps.forEach((step, index) => {
                    step.classList.toggle('active', index === currentStep);
                });
                currentStep++;
            } else {
                clearInterval(interval);
            }
        }, 1000);
    }

    hideMessages() {
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');
        
        if (errorMessage) errorMessage.style.display = 'none';
        if (successMessage) successMessage.style.display = 'none';
    }

    showSuccess(data) {
        // Hide the form
        const form = document.getElementById('analysisForm');
        if (form) form.style.display = 'none';

        // Show success info
        const successInfo = document.getElementById('successInfo');
        if (successInfo) {
            successInfo.style.display = 'block';
            
            // Set patient ID
            const patientIdDisplay = document.getElementById('patientIdDisplay');
            if (patientIdDisplay && data.patient_id) {
                patientIdDisplay.textContent = data.patient_id;
            }
        }

        addDebugInfo(`Case submitted successfully with Patient ID: ${data.patient_id}`);
    }
}

// Debug Manager
class DebugManager {
    constructor() {
        this.debugInfo = document.getElementById('debugInfo');
    }

    addInfo(message) {
        if (this.debugInfo) {
            const timestamp = new Date().toLocaleTimeString();
            this.debugInfo.innerHTML += `[${timestamp}] ${message}\n`;
            this.debugInfo.scrollTop = this.debugInfo.scrollHeight;
        }
    }
}

// Utility function for backward compatibility
function addDebugInfo(message) {
    if (window.debugManager) {
        window.debugManager.addInfo(message);
    }
}

// Initialize all managers when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize managers
    window.themeManager = new ThemeManager();
    window.uiManager = new UIManager();
    window.fileUploadManager = new FileUploadManager();
    window.patientSubmissionManager = new PatientSubmissionManager();
    window.debugManager = new DebugManager();

    // Test backend connection
    testBackendConnection();
});

// Test backend connection on page load
async function testBackendConnection() {
    addDebugInfo('Page loaded, testing backend connection...');
    
    try {
        const response = await fetch('http://localhost:5000/api/health');
        const data = await response.json();

        if (data.status === 'healthy') {
            addDebugInfo('Backend connection successful');
            uiManager.showNotification('Backend connected successfully!', 'success');
        } else {
            addDebugInfo(`Backend responded but not healthy: ${JSON.stringify(data)}`);
        }
    } catch (error) {
        addDebugInfo(`Backend connection failed: ${error.message}`);
        uiManager.showNotification('Backend server not connected. Please start the backend on port 5000.', 'error');
    }
}

// Add CSS for notifications
const notificationStyles = `
<style>
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 16px 20px;
    border-radius: 12px;
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 12px;
    z-index: 1001;
    transform: translateX(400px);
    transition: transform 0.3s ease;
    max-width: 350px;
}

.notification.show {
    transform: translateX(0);
}

.notification-success {
    border-color: var(--success-color);
    background: rgba(16, 185, 129, 0.1);
}

.notification-error {
    border-color: var(--error-color);
    background: rgba(239, 68, 68, 0.1);
}

.notification-warning {
    border-color: var(--warning-color);
    background: rgba(245, 158, 11, 0.1);
}

.notification-info {
    border-color: var(--accent-color);
    background: rgba(6, 182, 212, 0.1);
}

.notification i {
    font-size: 1.2rem;
}

.notification-success i {
    color: var(--success-color);
}

.notification-error i {
    color: var(--error-color);
}

.notification-warning i {
    color: var(--warning-color);
}

.notification-info i {
    color: var(--accent-color);
}

@keyframes slideOutDown {
    from {
        opacity: 1;
        transform: translateY(0);
    }
    to {
        opacity: 0;
        transform: translateY(30px);
    }
}

.animate-in {
    animation: fadeInUp 0.6s ease forwards;
}

.focused {
    transform: scale(1.02);
}

.focused textarea {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}
</style>
`;

// Inject notification styles
document.head.insertAdjacentHTML('beforeend', notificationStyles);

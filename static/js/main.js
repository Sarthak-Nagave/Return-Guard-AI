/**
 * Main JavaScript file for Return Abuse Detection System
 */

// Global variables
let chartInstances = {};

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize charts if on results page
    if (document.querySelector('.risk-gauge')) {
        initializeCharts();
    }
    
    // Initialize keyboard shortcuts
    initializeKeyboardShortcuts();
    
    // Initialize escalation button
    initializeEscalationButton();
    
    // Initialize auto-save functionality
    initializeAutoSave();
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Real-time validation
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', debounce(validateField, 300));
    });
}

/**
 * Validate individual form field
 */
function validateField(event) {
    const field = event.target;
    const fieldName = field.name;
    const value = field.value.trim();
    
    // Clear previous validation messages
    clearFieldError(field);
    
    // Validate based on field type
    switch(fieldName) {
        case 'user_id':
            validateUserId(field, value);
            break;
        case 'order_id':
            validateOrderId(field, value);
            break;
        case 'product_id':
            validateProductId(field, value);
            break;
        case 'return_message_sentiment':
            validateSentiment(field, value);
            break;
        case 'user_return_count':
            validateReturnCount(field, value);
            break;
        case 'return_approval_time':
            validateApprovalTime(field, value);
            break;
    }
}

/**
 * Validation functions
 */
function validateUserId(field, value) {
    if (!value) {
        setFieldError(field, 'User ID is required');
        return false;
    }
    if (!/^U\d{5}$/.test(value)) {
        setFieldError(field, 'User ID must be in format U00000');
        return false;
    }
    setFieldSuccess(field);
    return true;
}

function validateOrderId(field, value) {
    if (!value) {
        setFieldError(field, 'Order ID is required');
        return false;
    }
    if (!/^O\d{6}$/.test(value)) {
        setFieldError(field, 'Order ID must be in format O000000');
        return false;
    }
    setFieldSuccess(field);
    return true;
}

function validateProductId(field, value) {
    if (!value) {
        setFieldError(field, 'Product ID is required');
        return false;
    }
    if (!/^P\d{5}$/.test(value)) {
        setFieldError(field, 'Product ID must be in format P00000');
        return false;
    }
    setFieldSuccess(field);
    return true;
}

function validateSentiment(field, value) {
    const sentiment = parseFloat(value);
    if (isNaN(sentiment)) {
        setFieldError(field, 'Sentiment must be a valid number');
        return false;
    }
    if (sentiment < -1 || sentiment > 1) {
        setFieldError(field, 'Sentiment must be between -1 and 1');
        return false;
    }
    setFieldSuccess(field);
    return true;
}

function validateReturnCount(field, value) {
    const count = parseInt(value);
    if (isNaN(count) || count < 0) {
        setFieldError(field, 'Return count must be a non-negative integer');
        return false;
    }
    if (count > 50) {
        setFieldWarning(field, 'Unusually high return count - please verify');
    } else {
        setFieldSuccess(field);
    }
    return true;
}

function validateApprovalTime(field, value) {
    const time = parseInt(value);
    if (isNaN(time) || time < 0) {
        setFieldError(field, 'Approval time must be a non-negative integer');
        return false;
    }
    if (time > 30) {
        setFieldWarning(field, 'Approval time seems unusually long');
    } else {
        setFieldSuccess(field);
    }
    return true;
}

/**
 * Field validation UI helpers
 */
function setFieldError(field, message) {
    field.classList.add('is-invalid');
    field.classList.remove('is-valid');
    
    let feedback = field.parentNode.querySelector('.invalid-feedback');
    if (!feedback) {
        feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        field.parentNode.appendChild(feedback);
    }
    feedback.textContent = message;
}

function setFieldSuccess(field) {
    field.classList.add('is-valid');
    field.classList.remove('is-invalid');
    clearFieldError(field);
}

function setFieldWarning(field, message) {
    field.classList.remove('is-invalid', 'is-valid');
    
    let feedback = field.parentNode.querySelector('.warning-feedback');
    if (!feedback) {
        feedback = document.createElement('div');
        feedback.className = 'warning-feedback text-warning small';
        field.parentNode.appendChild(feedback);
    }
    feedback.textContent = message;
}

function clearFieldError(field) {
    field.classList.remove('is-invalid', 'is-valid');
    
    const feedbacks = field.parentNode.querySelectorAll('.invalid-feedback, .warning-feedback');
    feedbacks.forEach(feedback => feedback.remove());
}

/**
 * Initialize charts and visualizations
 */
function initializeCharts() {
    // Risk distribution chart
    createRiskDistributionChart();
    
    // Animated progress bars
    animateProgressBars();
}

/**
 * Create risk distribution chart
 */
function createRiskDistributionChart() {
    const canvas = document.getElementById('riskChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    chartInstances.riskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [60, 30, 10],
                backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

/**
 * Animate progress bars
 */
function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
}

/**
 * Initialize keyboard shortcuts
 */
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + Enter to submit form
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            const form = document.querySelector('form');
            if (form) {
                form.submit();
            }
        }
        
        // Ctrl/Cmd + R to reset form
        if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
            event.preventDefault();
            const resetBtn = document.querySelector('button[type="reset"]');
            if (resetBtn) {
                resetBtn.click();
            }
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal.show');
            openModals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
    });
}

/**
 * Initialize auto-save functionality
 */
function initializeAutoSave() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input, select, textarea');
    
    inputs.forEach(input => {
        input.addEventListener('change', debounce(saveFormData, 1000));
    });
    
    // Load saved data on page load
    loadFormData();
}

/**
 * Save form data to localStorage
 */
function saveFormData() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    localStorage.setItem('returnFormData', JSON.stringify(data));
}

/**
 * Load form data from localStorage
 */
function loadFormData() {
    const savedData = localStorage.getItem('returnFormData');
    if (!savedData) return;
    
    try {
        const data = JSON.parse(savedData);
        
        Object.entries(data).forEach(([key, value]) => {
            const field = document.querySelector(`[name="${key}"]`);
            if (field) {
                field.value = value;
            }
        });
    } catch (error) {
        console.error('Error loading saved form data:', error);
    }
}

/**
 * Clear saved form data
 */
function clearSavedData() {
    localStorage.removeItem('returnFormData');
}

/**
 * Utility function: Debounce
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    alertDiv.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

/**
 * Get icon for notification type
 */
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Export functions for global access
 */
window.ReturnAbuseApp = {
    showNotification,
    clearSavedData,
    validateField,
    loadSampleData: function() {
        // This function is called from the HTML template
        document.getElementById('user_id').value = 'U00467';
        document.getElementById('order_id').value = 'O000010';
        document.getElementById('product_id').value = 'P01821';
        document.getElementById('return_reason').value = 'wrong item delivered';
        document.getElementById('image_uploaded').value = '1';
        document.getElementById('user_return_count').value = '1';
        document.getElementById('return_message_sentiment').value = '-0.72';
        document.getElementById('return_approval_time').value = '11';
        
        showNotification('Sample data loaded successfully!', 'success');
    }
};

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred. Please try again.', 'danger');
});

// Escalation button handler
function initializeEscalationButton() {
    const escalateBtn = document.getElementById('escalate-btn');
    if (escalateBtn) {
        escalateBtn.addEventListener('click', handleEscalation);
    }
}

function handleEscalation() {
    const escalateBtn = document.getElementById('escalate-btn');
    
    // Get case data from the results page
    const caseData = {
        user_id: document.querySelector('[data-user-id]')?.getAttribute('data-user-id') || '',
        order_id: document.querySelector('[data-order-id]')?.getAttribute('data-order-id') || '',
        product_id: document.querySelector('[data-product-id]')?.getAttribute('data-product-id') || '',
        return_reason: document.querySelector('[data-return-reason]')?.getAttribute('data-return-reason') || '',
        abuse_probability: document.querySelector('[data-abuse-probability]')?.getAttribute('data-abuse-probability') || '',
        risk_score: document.querySelector('[data-risk-score]')?.getAttribute('data-risk-score') || ''
    };
    
    // Disable button during request
    escalateBtn.disabled = true;
    escalateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Escalating...';
    
    // Send escalation request
    fetch('/escalate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(caseData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(data.message, 'success');
            escalateBtn.innerHTML = '<i class="fas fa-check"></i> Escalated';
            escalateBtn.classList.remove('btn-warning');
            escalateBtn.classList.add('btn-success');
        } else {
            showNotification(data.error || 'Failed to escalate case', 'danger');
            escalateBtn.disabled = false;
            escalateBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Escalate to Manager';
        }
    })
    .catch(error => {
        console.error('Escalation error:', error);
        showNotification('Failed to escalate case. Please try again.', 'danger');
        escalateBtn.disabled = false;
        escalateBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Escalate to Manager';
    });
}

// Service worker registration for offline support
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(error) {
                console.log('ServiceWorker registration failed');
            });
    });
}

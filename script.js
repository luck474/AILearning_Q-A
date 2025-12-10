
// State
let currentQuestionIndex = 0;
let score = 0;
let userAnswers = new Array(questions.length).fill(null);

// DOM Elements
const qIdEl = document.getElementById('q-id');
const qTextEl = document.getElementById('question-text');
const optionsContainer = document.getElementById('options-container');
const progressText = document.getElementById('progress-text');
const progressFill = document.getElementById('progress-fill');
const btnNext = document.getElementById('btn-next');
const btnPrev = document.getElementById('btn-prev');
const feedbackEl = document.getElementById('feedback');
const qTypeBadge = document.getElementById('question-type');
const questionGrid = document.getElementById('question-grid');

// Initialize
function init() {
    renderGrid();
    loadQuestion(currentQuestionIndex);
    updateProgress();
    initMobileSidebar();

    btnNext.addEventListener('click', () => {
        if (currentQuestionIndex < questions.length - 1) {
            currentQuestionIndex++;
            loadQuestion(currentQuestionIndex);
            updateProgress();
        }
    });

    btnPrev.addEventListener('click', () => {
        if (currentQuestionIndex > 0) {
            currentQuestionIndex--;
            loadQuestion(currentQuestionIndex);
            updateProgress();
        }
    });
}

// Mobile Sidebar Toggle
function initMobileSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const sidebarTitle = sidebar.querySelector('h3');

    // Check if mobile
    function isMobile() {
        return window.innerWidth <= 768;
    }

    // Initialize collapsed state on mobile
    if (isMobile()) {
        sidebar.classList.add('collapsed');
    }

    // Toggle on click
    sidebarTitle.addEventListener('click', () => {
        if (isMobile()) {
            sidebar.classList.toggle('collapsed');
        }
    });

    // Handle resize
    window.addEventListener('resize', () => {
        if (isMobile()) {
            if (!sidebar.classList.contains('collapsed')) {
                // Keep expanded state if user manually expanded
            }
        } else {
            sidebar.classList.remove('collapsed');
        }
    });
}

function loadQuestion(index) {
    const q = questions[index];

    // Update Header
    qIdEl.textContent = q.id;
    // Determine Display Type
    let typeText = '选择题 Multiple Choice';
    if (q.type === 'true_false') typeText = '判断题 True/False';
    else if (q.type === 'multiple_response') typeText = '多选题 Multiple Response';
    qTypeBadge.textContent = typeText;

    // Animate Card (Simple reset)
    const card = document.querySelector('.card');
    card.style.animation = 'none';
    card.offsetHeight; /* trigger reflow */
    card.style.animation = 'fadeIn 0.4s ease-out';

    // Set Text
    qTextEl.textContent = q.question;

    // Clear Options
    optionsContainer.innerHTML = '';
    feedbackEl.innerHTML = '';
    feedbackEl.className = 'feedback-area';

    // Reset Multi Selections
    currentMultiSelections = new Set();

    // Create Options
    q.options.forEach((opt, idx) => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.textContent = opt;

        btn.onclick = () => handleSelect(btn, opt, q);
        optionsContainer.appendChild(btn);
    });

    // Add Submit Button for Multiple Response
    if (q.type === 'multiple_response' && !userAnswers[index]) {
        const submitBtn = document.createElement('button');
        submitBtn.id = 'btn-submit';
        submitBtn.className = 'btn btn-primary submit-btn';
        submitBtn.textContent = 'Submit Answer';
        submitBtn.disabled = true; // Disabled until at least one option selected
        submitBtn.onclick = () => submitMultipleResponse(q);
        optionsContainer.appendChild(submitBtn);
    }

    // Handle restoration of previous answer
    const savedAnswer = userAnswers[index];
    if (savedAnswer) {
        // Restore visuals
        if (q.type === 'multiple_response') {
            // savedAnswer is an array of selected option strings
            const buttons = optionsContainer.querySelectorAll('.option-btn');
            buttons.forEach(b => {
                if (savedAnswer.includes(b.textContent)) {
                    // Check visuals for this button
                    // For multiple response, we check if it is part of the correct set or wrong set
                    // But checkAnswerVisuals is designed for single select or iterating.
                    // We will use a dedicated visualizer for simple restoration or reuse checkAnswerVisuals logic
                }
            });
            // Actually, checkAnswerVisuals can handle the logic if we pass the full selection
            checkAnswerVisuals(null, savedAnswer, q.answer, true, q.type);
        } else {
            const buttons = optionsContainer.querySelectorAll('.option-btn');
            let selectedBtn = null;
            buttons.forEach(b => {
                if (b.textContent === savedAnswer) {
                    selectedBtn = b;
                }
            });
            if (selectedBtn) {
                checkAnswerVisuals(selectedBtn, savedAnswer, q.answer, true, q.type);
            }
        }
    }

    // Update Buttons
    btnPrev.disabled = index === 0;
    btnNext.innerHTML = index === questions.length - 1 ? 'Finish' : `Next <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>`;
}

// Track current selections for multiple response
let currentMultiSelections = new Set();

function handleSelect(btn, selectedOption, q) {
    // If already answered, ignore
    if (userAnswers[currentQuestionIndex]) return;

    if (q.type === 'multiple_response') {
        // Toggle selection
        if (currentMultiSelections.has(selectedOption)) {
            currentMultiSelections.delete(selectedOption);
            btn.classList.remove('selected');
        } else {
            currentMultiSelections.add(selectedOption);
            btn.classList.add('selected');
        }

        // Update Submit button
        const submitBtn = document.getElementById('btn-submit');
        if (submitBtn) {
            submitBtn.disabled = currentMultiSelections.size === 0;
        }
    } else {
        // Single select (immediate submit)
        submitSingleResponse(btn, selectedOption, q);
    }
}

function submitSingleResponse(btn, selectedOption, q) {
    userAnswers[currentQuestionIndex] = selectedOption;

    let isCorrect = checkCorrectness(selectedOption, q.answer, q.type);
    checkAnswerVisuals(btn, selectedOption, q.answer, false, q.type);
    updateGridStatus(currentQuestionIndex, isCorrect);
}

function submitMultipleResponse(q) {
    const selectedOptions = Array.from(currentMultiSelections);
    userAnswers[currentQuestionIndex] = selectedOptions; // Store array

    // Check correctness
    // Construct selected key string (e.g. "AB")
    // Extract keys
    let selectedKeys = selectedOptions.map(opt => {
        const match = opt.match(/^([A-D])[\.、]/);
        return match ? match[1] : '';
    }).filter(k => k).sort().join('');

    // Correct key is "ABCD"
    const isCorrect = selectedKeys === q.answer;

    checkAnswerVisuals(null, selectedOptions, q.answer, false, q.type);
    updateGridStatus(currentQuestionIndex, isCorrect);
}

function checkCorrectness(selected, answer, type) {
    if (type === 'true_false') {
        return selected.toUpperCase() === answer.toString().toUpperCase();
    } else if (type === 'multiple_response') {
        // handled in submitMultipleResponse usually, but helper if needed
        return false;
    } else {
        // Single choice
        const match = selected.match(/^([A-D])[\.、]/);
        let key = match ? match[1] : selected;
        return key === answer;
    }
}

function checkAnswerVisuals(selectedBtn, selectedValue, correctKey, isRestoring, type) {
    // Disable all buttons
    const buttons = optionsContainer.querySelectorAll('.option-btn');
    buttons.forEach(b => {
        b.classList.add('disabled');
        b.onclick = null;
    });

    // Remove submit button if exists
    const submitBtn = document.getElementById('btn-submit');
    if (submitBtn) submitBtn.remove();

    if (type === 'multiple_response') {
        // selectedValue is Array of strings
        const selectedSet = new Set(selectedValue);

        buttons.forEach(b => {
            const match = b.textContent.match(/^([A-D])[\.、]/);
            const key = match ? match[1] : '';

            const isSelected = selectedSet.has(b.textContent);
            const isCorrectKey = correctKey.includes(key); // correctKey is "ABCD"

            if (isSelected) {
                if (isCorrectKey) {
                    b.classList.add('correct');
                } else {
                    b.classList.add('wrong');
                }
            } else {
                if (isCorrectKey) {
                    b.classList.add('correct'); // Highlight missed correct answers
                }
            }
        });

    } else {
        // Single Select Logic
        // Re-determine correctness locally
        let isCorrect = checkCorrectness(selectedValue, correctKey, type);

        if (isCorrect) {
            if (selectedBtn) selectedBtn.classList.add('correct');
        } else {
            if (selectedBtn) selectedBtn.classList.add('wrong');
            // Highlight correct one
            buttons.forEach(b => {
                let key = "";
                if (type === 'true_false') {
                    key = b.textContent.toUpperCase();
                    if (key === correctKey.toString().toUpperCase()) b.classList.add('correct');
                } else {
                    const m = b.textContent.match(/^([A-D])[\.、]/);
                    key = m ? m[1] : b.textContent;
                    if (key === correctKey) b.classList.add('correct');
                }
            });
        }
    }
}

function updateProgress() {
    progressText.innerText = `${currentQuestionIndex + 1} / ${questions.length}`;
    const pct = ((currentQuestionIndex + 1) / questions.length) * 100;
    progressFill.style.width = `${pct}%`;
    updateGridHighlight();
}

function renderGrid() {
    questionGrid.innerHTML = '';
    questions.forEach((q, idx) => {
        const item = document.createElement('div');
        item.className = 'grid-item';
        item.textContent = idx + 1;
        item.id = `grid-item-${idx}`;
        item.onclick = () => {
            currentQuestionIndex = idx;
            loadQuestion(currentQuestionIndex);
            updateProgress();
        };
        questionGrid.appendChild(item);
    });
}

function updateGridHighlight() {
    // Remove active class from all
    const items = questionGrid.querySelectorAll('.grid-item');
    items.forEach(item => item.classList.remove('active'));

    // Add active to current
    const current = document.getElementById(`grid-item-${currentQuestionIndex}`);
    if (current) current.classList.add('active');
}

function updateGridStatus(index, isCorrect) {
    const item = document.getElementById(`grid-item-${index}`);
    if (item) {
        item.classList.remove('answered', 'wrong');
        item.classList.add(isCorrect ? 'answered' : 'wrong');
    }
}

init();

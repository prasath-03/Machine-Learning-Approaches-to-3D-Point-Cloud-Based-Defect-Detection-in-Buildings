// BuildScan AI JavaScript Functions

// Function to animate counters
function animateCounter(element, target, duration) {
  const start = parseInt(element.textContent.replace(/,/g, ''));
  const increment = (target - start) / (duration / 16);
  let current = start;
  
  const timer = setInterval(() => {
    current += increment;
    if ((increment > 0 && current >= target) || (increment < 0 && current <= target)) {
      clearInterval(timer);
      current = target;
    }
    element.textContent = Math.floor(current).toLocaleString();
  }, 16);
}

// Function to handle tabs
function openTab(evt, tabName) {
  // Hide all tab content
  const tabcontent = document.getElementsByClassName("tab-content");
  for (let i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Remove active class from all tabs
  const tablinks = document.getElementsByClassName("tab-link");
  for (let i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the current tab and add active class to the button
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", function() {
  // Initialize counters
  const counterElements = document.querySelectorAll(".counter");
  counterElements.forEach(element => {
    const target = parseInt(element.getAttribute('data-target').replace(/,/g, ''));
    animateCounter(element, target, 2000);
  });

  // Set default tab
  if (document.querySelector('.tab-link')) {
    document.querySelector('.tab-link').click();
  }
});

// Toggle sidebar function
function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  sidebar.classList.toggle('collapsed');
}

// Function to toggle dark/light mode
function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
  const isDarkMode = document.body.classList.contains('dark-mode');
  localStorage.setItem('darkMode', isDarkMode);
}

// Check for saved dark mode preference
const savedDarkMode = localStorage.getItem('darkMode');
if (savedDarkMode === 'true') {
  document.body.classList.add('dark-mode');
}
const app = document.getElementById("app");

// Database Setup
const dbName = "ResultDB";
const storeName = "students";
let db;

// Open or create the IndexedDB database
function openDatabase() {
  const request = indexedDB.open(dbName, 1);
  
  request.onupgradeneeded = (event) => {
    db = event.target.result;
    if (!db.objectStoreNames.contains(storeName)) {
      const objectStore = db.createObjectStore(storeName, { keyPath: "id", autoIncrement: true });
      objectStore.createIndex("name", "name", { unique: false });
      objectStore.createIndex("subject", "subject", { unique: false });
    }
  };
  
  request.onsuccess = (event) => {
    db = event.target.result;
    loadFromIndexedDB();  // Load data after DB is ready
  };
  
  request.onerror = (event) => {
    console.error("Error opening database:", event.target.error);
  };
}

// Add student data to IndexedDB
function addStudent(student) {
  const transaction = db.transaction([storeName], "readwrite");
  const objectStore = transaction.objectStore(storeName);
  const request = objectStore.add(student);

  request.onsuccess = () => {
    console.log("Student added:", student);
  };

  request.onerror = (event) => {
    console.error("Error adding student:", event.target.error);
  };
}

// Get all student records from IndexedDB
function getStudents(callback) {
  const transaction = db.transaction([storeName], "readonly");
  const objectStore = transaction.objectStore(storeName);
  const request = objectStore.getAll();

  request.onsuccess = (event) => {
    callback(event.target.result);
  };

  request.onerror = (event) => {
    console.error("Error retrieving students:", event.target.error);
  };
}

// Calculate total and grade based on CA and Exam scores
function calculateResults(firstCA, secondCA, exam) {
  const total = firstCA + secondCA + exam;
  let grade;
  if (total >= 80) grade = "A";
  else if (total >= 70) grade = "B";
  else if (total >= 60) grade = "C";
  else if (total >= 50) grade = "D";
  else grade = "F";
  return { total, grade };
}

// Render the table of results
function renderTable(students) {
  const tableHTML = `
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Subject</th>
          <th>First C.A</th>
          <th>Second C.A</th>
          <th>Exam</th>
          <th>Total</th>
          <th>Grade</th>
        </tr>
      </thead>
      <tbody>
        ${students
          .map(
            (student) => `
          <tr>
            <td>${student.name}</td>
            <td>${student.subject}</td>
            <td>${student.scores.firstCA}</td>
            <td>${student.scores.secondCA}</td>
            <td>${student.scores.exam}</td>
            <td>${student.total}</td>
            <td>${student.grade}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
  return tableHTML;
}

// Render the input form
function renderForm() {
  return `
    <form id="student-form">
      <label for="file-upload">Upload Excel File:</label>
      <input type="file" id="file-upload" name="file-upload" accept=".xlsx, .xls" required>

      <button type="submit">Upload Results</button>
    </form>
  `;
}

// Handle Excel file upload and parsing
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    const data = e.target.result;
    const workbook = XLSX.read(data, { type: "binary" });
    const sheetName = workbook.SheetNames[0];  // Assuming data is in the first sheet
    const worksheet = workbook.Sheets[sheetName];
    
    // Parse the worksheet into JSON
    const studentsData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    processStudentData(studentsData);
  };
  reader.readAsBinaryString(file);
}

// Process the parsed student data
function processStudentData(data) {
  const subjects = data[0];  // First row is the subject names
  const studentRecords = [];

  // Process each student's data
  for (let i = 1; i < data.length; i++) {
    const studentRow = data[i];
    const student = {
      name: studentRow[0], // Assuming the name is in the first column
      subject: studentRow[1], // Subject in the second column
      scores: {
        firstCA: studentRow[2], // First C.A in the third column
        secondCA: studentRow[3], // Second C.A in the fourth column
        exam: studentRow[4], // Exam score in the fifth column
      },
    };

    const { total, grade } = calculateResults(student.scores.firstCA, student.scores.secondCA, student.scores.exam);
    student.total = total;
    student.grade = grade;

    studentRecords.push(student);
  }

  // Add each student record to IndexedDB
  studentRecords.forEach(addStudent);
  loadFromIndexedDB();
}

// Render the entire app
function renderApp() {
  app.innerHTML = `
    <div class="container">
      <h1>Result Management System</h1>
      ${renderForm()}
      ${students.length > 0 ? renderTable(students) : "<p>No results yet. Add a result above.</p>"}
    </div>
  `;

  const form = document.getElementById("student-form");
  const fileInput = document.getElementById("file-upload");

  fileInput.addEventListener("change", handleFileUpload);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadFromIndexedDB();  // Refresh table after uploading
  });
}

// Load data from IndexedDB and render it
function loadFromIndexedDB() {
  getStudents((storedStudents) => {
    students = storedStudents;
    renderApp();
  });
}

// Initialize the database
openDatabase();

// Initial render
renderApp();

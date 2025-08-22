// List of JSON files
const jsonFiles = [
  "/json/mfr.json"
];

const tableBody = document.querySelector("#product-data tbody");

// Store merged data by ID
const mergedData = {};

// Load all files, then build table
async function loadData() {
  for (const file of jsonFiles) {
    try {
      const response = await fetch(file);
      const data = await response.json();

      data.forEach(item => {
        const id = item.id;

        // Ensure object exists
        if (!mergedData[id]) {
          mergedData[id] = { id };
        }

        // Merge properties
        Object.assign(mergedData[id], item);
      });
    } catch (err) {
      console.error(`Error loading ${file}:`, err);
    }
  }

  buildTable();
}

// Build the HTML table
function buildTable() {
  Object.values(mergedData).forEach(row => {
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td>${row.id ?? ""}</td>
      <td>${row.manufacturer ?? ""}</td>
    `;

    tableBody.appendChild(tr);
  });
}

loadData();

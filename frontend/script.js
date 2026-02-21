async function analyzeResume() {

  const fileInput = document.getElementById("resumeFile")
  const domain = document.getElementById("domainSelect").value

  if (!fileInput.files.length) {
    alert("Upload resume")
    return
  }

  if (!domain) {
    alert("Select domain")
    return
  }

  const formData = new FormData()
  formData.append("file", fileInput.files[0])
  formData.append("selected_domain", domain)

  const response = await fetch(
    "https://resume-domain-classifier.onrender.com/analyze",
    {
      method: "POST",
      body: formData
    }
  )

  const data = await response.json()

  document.getElementById("results").style.display = "flex"

  document.getElementById("skillMatch").innerText = data.skill_match
  document.getElementById("confidence").innerText = data.model_confidence
  document.getElementById("finalScore").innerText = data.confidence

  document.getElementById("details").innerHTML = `
    <p><b>Selected:</b> ${data.selected_domain}</p>
    <p><b>Final:</b> ${data.final_domain}</p>
    <p><b>Decision:</b> ${data.decision}</p>
  `
}
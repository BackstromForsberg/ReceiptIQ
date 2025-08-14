// app/actions.ts
"use server";

const API = process.env.API_BASE_URL ?? "http://api:5000"; // fallback

export async function getHello() {
  try {
    const res = await fetch(`${API}/hello`, { cache: "no-store" });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status} ${res.statusText} ${text}`);
    }
    return res.text();
  } catch (err: any) {
    // Helpful diagnostics
    console.error("Fetch to API failed:", {
      message: err?.message,
      cause: err?.cause,
      url: `${API}/hello`,
    });
    throw err;
  }
}

export async function sendPDFReceiptForOCR(
  arrayBuffer: ArrayBuffer,
  filename = "upload.jpg",
  type: "pdf" | "png" = "pdf" // Default to PDF, can be PNG
): Promise<any> {
  try {
    const response = await fetch(`${API}/ocr`, {
      method: "POST",
      headers: {
        "Content-Type": type === "png" ? "image/png" : "application/pdf", // Different if PNG
        "X-Filename": filename, // Optional, passed to backend for metadata
      },
      body: arrayBuffer,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status} ${errorText}`);
    }

    const result = await response.json();
    console.log("OCR result:", result);
    return result;
  } catch (err) {
    console.error("Error sending receipt:", err);
    throw err;
  }
}

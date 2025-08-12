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

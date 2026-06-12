async function request(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (Array.isArray(data.detail)) {
        detail = data.detail.map((entry) => entry.msg || JSON.stringify(entry)).join("; ");
      } else if (data.detail) {
        detail = data.detail;
      }
    } catch (_) { /* not JSON */ }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const api = {
  get: (url) => request("GET", url),
  post: (url, body = {}) => request("POST", url, body),
  del: (url) => request("DELETE", url),
  async urdf() {
    const response = await fetch("/api/urdf");
    if (!response.ok) throw new Error(`URDF fetch failed: ${response.status}`);
    return { text: await response.text(), source: response.headers.get("X-Urdf-Source") || "?" };
  },
};

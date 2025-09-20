import { Hono } from "hono";

type Bindings = {
  MODAL_SIGLIP_ENDPOINT: string;
  MODAL_SIGLIP_HEALTH_ENDPOINT: string;
  MODAL_COLQWEN_ENDPOINT: string;
  MODAL_COLQWEN_HEALTH_ENDPOINT: string;
  MODAL_QWEN_ENDPOINT: string;
  MODAL_QWEN_HEALTH_ENDPOINT: string;
};

const app = new Hono<{ Bindings: Bindings }>();

app.get("/", (c) => {
  return c.json({
    name: "Masala Embed API",
    endpoints: [
      "/health",
      "/embedding/siglip",
      "/embedding/colqwen",
      "/embedding/qwen",
    ],
  });
});

app.get("/health", async (c) => {
  const healthEndpoints = [
    c.env?.MODAL_SIGLIP_HEALTH_ENDPOINT,
    c.env?.MODAL_COLQWEN_HEALTH_ENDPOINT,
    c.env?.MODAL_QWEN_HEALTH_ENDPOINT,
  ];

  const results = await Promise.allSettled(
    healthEndpoints.map(async (endpoint) => {
      if (!endpoint) return false;
      try {
        await fetch(endpoint);
        return true;
      } catch {
        return false;
      }
    }),
  );

  return c.json({
    siglip: results[0].status === "fulfilled" ? results[0].value : false,
    colqwen: results[1].status === "fulfilled" ? results[1].value : false,
    qwen: results[2].status === "fulfilled" ? results[2].value : false,
  });
});

app.post("/embedding/siglip", async (c) => {
  const data = await c.req.json();

  if (!c.env?.MODAL_SIGLIP_ENDPOINT) {
    return c.json({ error: "SigLIP endpoint not configured" }, 500);
  }

  try {
    const response = await fetch(c.env.MODAL_SIGLIP_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = (await response.json()) as { embedding: number[] };
    return c.json({
      embedding: result.embedding,
      model: "siglip",
    });
  } catch (error) {
    return c.json({ error: "Request failed" }, 500);
  }
});

app.post("/embedding/colqwen", async (c) => {
  const data = await c.req.json();

  if (!c.env?.MODAL_COLQWEN_ENDPOINT) {
    return c.json({ error: "ColQwen endpoint not configured" }, 500);
  }

  try {
    const response = await fetch(c.env.MODAL_COLQWEN_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = (await response.json()) as { embedding: number[] };
    return c.json({
      embedding: result.embedding,
      model: "colqwen",
    });
  } catch (error) {
    return c.json({ error: "Request failed" }, 500);
  }
});

app.post("/embedding/qwen", async (c) => {
  const data = await c.req.json();

  if (!c.env?.MODAL_QWEN_ENDPOINT) {
    return c.json({ error: "Qwen endpoint not configured" }, 500);
  }

  try {
    const response = await fetch(c.env.MODAL_QWEN_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = (await response.json()) as { embedding: number[] };
    return c.json({
      embedding: result.embedding,
      model: "qwen",
    });
  } catch (error) {
    return c.json({ error: "Request failed" }, 500);
  }
});

export default app;

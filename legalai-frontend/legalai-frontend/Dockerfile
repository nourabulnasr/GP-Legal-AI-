# LegalAI frontend – Vite dev server (connects to backend at VITE_API_BASE_URL)
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Browser will call backend at this URL (host's localhost when using docker compose)
ENV VITE_API_BASE_URL=http://localhost:8000

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

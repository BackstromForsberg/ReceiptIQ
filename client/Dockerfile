# Use Node.js LTS
FROM node:22-bookworm-slim

# Environment variables for dev
ENV NODE_ENV=development \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

# Install dependencies first (for caching)
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the app
COPY . .

EXPOSE 3000

# Start Next.js in dev mode with hot reload
CMD ["npm", "run", "dev", "--", "-p", "3000", "-H", "0.0.0.0"]
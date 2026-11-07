@echo off
rem Starts the OTSR web app from the prebuilt dist/ folder and opens it in the default browser.
rem Needs Node.js 20 or newer (https://nodejs.org). Close this window to stop the server.
cd /d "%~dp0"
where node >nul 2>nul || (echo Node.js was not found. Install it from https://nodejs.org and run this file again. & pause & exit /b 1)
if not exist dist\index.html (
  echo dist\index.html is missing. Building it needs the dev dependencies: npm install ^&^& npm run build
  pause
  exit /b 1
)
node tools\serve.mjs %*

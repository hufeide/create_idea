import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = 10000;

function sendFile(req, res, filePath) {
  const ext = path.extname(filePath);
  let contentType = 'text/plain';
  
  if (ext === '.html') contentType = 'text/html';
  else if (ext === '.js') contentType = 'application/javascript';
  else if (ext === '.json') contentType = 'application/json';
  else if (ext === '.css') contentType = 'text/css';
  
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(500);
      res.end('Error');
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  let filePath;
  
  if (req.url === '/' || req.url === '/index.html') {
    filePath = path.join(__dirname, 'index.html');
  } else if (req.url === '/src/weather-card.js') {
    filePath = path.join(__dirname, 'src/weather-card.js');
  } else if (req.url === '/data/sample.json') {
    filePath = path.join(__dirname, 'data/sample.json');
  } else if (req.url === '/src/') {
    res.writeHead(404);
    res.end('Not Found');
    return;
  } else {
    res.writeHead(404);
    res.end('Not Found');
    return;
  }
  
  sendFile(req, res, filePath);
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});

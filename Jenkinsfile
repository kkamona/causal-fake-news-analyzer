// Equivalent of .github/workflows/ci.yml for Jenkins (declarative pipeline).
// Requires a Jenkins agent with Docker installed.
pipeline {
    agent any
    stages {
        stage('Unit tests') {
            agent { docker { image 'python:3.11-slim' } }
            steps {
                sh 'pip install --user -r requirements-dev.txt'
                sh 'python -m ruff check .'
                sh 'python -m pytest --cov=causalnews'
            }
        }
        stage('Build image') {
            steps { sh 'docker build -t causalnews:${BUILD_NUMBER} .' }
        }
        stage('Smoke test') {
            steps {
                sh '''
                  docker run -d --name causalnews-${BUILD_NUMBER} -p 8000:8000 causalnews:${BUILD_NUMBER}
                  sleep 10
                  curl -fs http://localhost:8000/health
                '''
            }
            post { always { sh 'docker rm -f causalnews-${BUILD_NUMBER} || true' } }
        }
    }
}



```markdown
# CSC-FUTO AI Academic Advisor

## Description

This is a chatbot that allows students to
interact with an academic advisor AI.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Amalagu/ai_academic_advisor.git
   ```

2. Navigate to the project directory:

   ```bash
   cd chatbot
   ```



## Configuration

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

3. Set up environment variables:

   Create a `.env` file in the project root and add the following:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   # Add any other environment variables here
   ```

## Usage

1. Run migrations:

   ```bash
   python manage.py migrate
   ```

2. Start the development server:

   ```bash
   python manage.py runserver
   ```

3. Open the application in your browser at [http://localhost:8000](http://localhost:8000).

## Contributing

If you would like to contribute to the project, follow these steps:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature_branch_name
   ```

3. Make your changes and commit:

   ```bash
   git commit -m "Your meaningful commit message"
   ```

4. Push to the branch:

   ```bash
   git push origin feature_branch_name
   ```

5. Open a pull request.

## License


.
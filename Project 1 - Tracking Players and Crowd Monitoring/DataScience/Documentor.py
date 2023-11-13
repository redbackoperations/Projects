import os

def process_docstring(docstring):
    # Basic conversion for some Sphinx-style tags
    docstring = docstring.replace(":param ", "**Parameter:** ").replace(":return:", "**Returns:**")
    return docstring.strip()

def py_to_md(file_path):
    markdown_output = ''
    inside_docstring = False
    docstring_content = ''
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            stripped_line = line.strip()
            # Handle multiline docstring
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                if inside_docstring:
                    inside_docstring = False
                    markdown_output += process_docstring(docstring_content) + '\n\n'
                    docstring_content = ''
                else:
                    inside_docstring = True
                    docstring_content += stripped_line[3:] + ' '
                continue
            if inside_docstring:
                docstring_content += stripped_line + ' '
                continue
            
            # Split the line into code and comment if an inline comment exists
            code_and_comment = line.split('#', 1)
            code_part = code_and_comment[0].strip()
            comment_part = code_and_comment[1].strip() if len(code_and_comment) > 1 else None

            if code_part:
                markdown_output += '```python\n' + code_part + '\n```\n\n'
            if comment_part:
                markdown_output += '*' + comment_part + '*\n\n'
    
    return markdown_output

def process_directory(directory_path, output_file):
    for root, dirs, files in os.walk(directory_path):
        output_file.write('# ' + os.path.basename(root) + '\n\n')  # Use directory name as a separator
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                markdown_content = py_to_md(file_path)
                output_file.write(markdown_content)

# Save the markdown content into a new .md file
with open('Model Documentation.md', 'w') as file:
    process_directory(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models', file)

print("Conversion completed!")

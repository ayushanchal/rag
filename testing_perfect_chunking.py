import re

def process_qa_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fixed the regex syntax error here:
    clean_content = re.sub(r'\\', '', content)
    
    # Split content into lines for processing
    lines = clean_content.split('\n')
    
    qa_pairs = []
    current_block = []
    
    # Pattern to identify questions or start of a QA block
    question_start_pattern = re.compile(
        r'^(What|Why|How|In Python|In R|Is |Can |Does |When |Where |Should |'
        r'Please explain|Important question|I have trouble|I understand|'
        r'I was wondering|I cannot import|We predicted|Since D2| b0 is|'
        r'To understand the P-value)', re.IGNORECASE
    )
    
    # Pattern to ignore page headers, footers, and section titles
    ignore_pattern = re.compile(r'^(Page \d+ of \d+|Machine Learning A-Z Q&A|\d+ Part \d+|\d+\.\d+ )', re.IGNORECASE)

    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines or metadata/headers
        if not line_stripped or ignore_pattern.match(line_stripped):
            continue
            
        # If we hit a new question, save the previous block if it exists
        if question_start_pattern.match(line_stripped):
            if current_block:
                qa_pairs.append("\n".join(current_block))
                current_block = []
        
        current_block.append(line)
        
    # Append the final block remaining
    if current_block:
        qa_pairs.append("\n".join(current_block))

    # Join all identified QA blocks using 6 newline characters (\n\n\n\n\n\n)
    separator = '\n\n\n\n\n\n'
    output_text = separator.join(qa_pairs)
    
    # Write to the new text file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
        
    print(f"Successfully processed {len(qa_pairs)} QA blocks and saved to {output_file_path}")

# Run the script (Make sure 'document.txt' is in your d:\AI_Engineering\rag\ folder)
process_qa_file('document.txt', 'separated_qa_output.txt')
import os
import glob
import json
import fitz
import pdfplumber
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_bulletin_paths = [
    r"dataset\\campus_facilities.pdf",
    r"dataset\\course_description.pdf",
    r"dataset\\holidays.pdf",
]

def extract_text_from_columns(pdf_bulletin_paths):
    all_extracted_texts = []
    
    # Loop over each PDF bulletin file path
    for pdf_bulletin_path in pdf_bulletin_paths:
        doc = fitz.open(pdf_bulletin_path)
        extracted_text = ""
    
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            width, height = page.rect.width, page.rect.height
            
            column1_rect = fitz.Rect(0, 0, width / 2, height)
            column2_rect = fitz.Rect(width / 2, 0, width, height)
            column1_text = page.get_text("text", clip=column1_rect)
            column2_text = page.get_text("text", clip=column2_rect)
            
            extracted_text += column1_text + "\n" + column2_text + "\n"
        all_extracted_texts.append(extracted_text)
    return all_extracted_texts


#print(extract_text_from_columns(pdf_bulletin_paths))

pdf_paths  = [
    
    "dataset\\list_of_courses.pdf",
    "dataset\\SU_constitution.pdf",
    "dataset\\details_of_programmes.pdf",
    "dataset\\WITW24.pdf",
    "dataset\\BITS_contacts.pdf",
]



def extract_content_pdfplumber(pdf_paths):
    all_content = []
    
    # Loop over each PDF file
    for pdf_path in pdf_paths:
        combined_data = ""  
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                combined_data += page_text
                
                tables_on_page = page.extract_tables()
                for table in tables_on_page:
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [' '.join(cell.splitlines()) if cell else '' for cell in row]
                        cleaned_table.append(cleaned_row)
                    
                    table_str = "\n".join([" - ".join(row) for row in cleaned_table])#convert the cleaned table to string with - as column separator
                    combined_data += "\n" + table_str + "\n"
        
        all_content.append(combined_data)
    
    return all_content

#print(extract_content_pdfplumber(pdf_paths))

handouts_folder_path = "dataset//handouts"
handouts_pdfs = glob.glob(os.path.join(handouts_folder_path, "*.pdf"))


def chunk_text_bypass(data_list):
    # Instead of chunking, return the content as-is, wrapped in a single "chunk"
    return [{"page_content": data} for data in data_list]

def extract_content_handouts(handout_pdfs, bypass_chunking=True):
    all_content = []
    
    # Loop over each PDF file
    for pdf_path in handout_pdfs:
        combined_data = ""  # For each file, initialize a string to hold combined data
        
        # Open the PDF with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Loop through all pages in the PDF
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                combined_data += page_text
                
                # Extract tables and clean up multi-line cells
                tables_on_page = page.extract_tables()
                for table in tables_on_page:
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [' '.join(cell.splitlines()) if cell else '' for cell in row]
                        cleaned_table.append(cleaned_row)
                    # Convert the cleaned table to string with "-" as column separator
                    table_str = "\n".join([" - ".join(row) for row in cleaned_table])
                    combined_data += "\n" + table_str + "\n"
        
        # Append the entire combined content for each PDF to all_content list
        all_content.append(combined_data)
    
    # Apply the bypass chunking function if bypass_chunking is True
    if bypass_chunking:
        return chunk_text_bypass(all_content)

result = extract_content_handouts(handouts_pdfs)
#print(result)



timetable_path = r"dataset\\timetable.json"
with open(timetable_path, 'r') as f:
    timetable_data = json.load(f)

def process_course_data(course_data):
    chunks = []
    for course_no, course_info in course_data.items():
        # Course basic information
        course_details = f"Course No: {course_no}\nCourse Name: {course_info['course_name']}\nUnits: {course_info['units']}"
        
        # Extract section details
        for section, section_info in course_info['sections'].items():
            instructors = ', '.join(section_info['instructor'])
            schedule_details = []
            for schedule in section_info['schedule']:
                days = ', '.join(schedule['days'])
                hours = ', '.join(str(hour) for hour in schedule['hours'])
                schedule_details.append(f"Room: {schedule['room']}, Days: {days}, Hours: {hours}")
            schedule_text = '\n'.join(schedule_details)
            
            section_details = f"\nSection: {section}\nInstructors: {instructors}\nSchedule:\n{schedule_text}"
            course_details += section_details

        # Extract exam details if available
        if 'exams' in course_info:
            exam_details = course_info['exams'][0]  # Assuming one set of exam details
            exam_text = f"Midsem: {exam_details['midsem']}, Compre: {exam_details['compre']}"
            course_details += f"\nExams:\n{exam_text}"
        
        # Combine all course details into one chunk
        chunks.append(course_details)
    
    return chunks


# Process the timetable data into text chunks
all_text_chunks = process_course_data(timetable_data["courses"])


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300, length_function=len)

def create_documents_with_labels():
    all_text_chunks = []
    bulletin_texts = extract_text_from_columns(pdf_bulletin_paths)
    for i, text in enumerate(bulletin_texts):
        # Split the text into chunks
        split_text_chunks = text_splitter.split_text(text)
        # Label each chunk and add to the list
        all_text_chunks += [{"label": f"bulletin_{i}_chunk_{j}", "content": chunk} for j, chunk in enumerate(split_text_chunks)]
    
    # Process and split content from PDF paths
    content_pdf = extract_content_pdfplumber(pdf_paths)
    for i, text in enumerate(content_pdf):
        split_text_chunks = text_splitter.split_text(text)
        all_text_chunks += [{"label": f"pdf_content_{i}_chunk_{j}", "content": chunk} for j, chunk in enumerate(split_text_chunks)]
    
    # Process handouts content without splitting
    content_handouts = extract_content_handouts(handouts_pdfs, bypass_chunking=True)
    all_text_chunks += [{"label": f"handout_{i}", "content": text["page_content"]} for i, text in enumerate(content_handouts)]
    
    # Process and split course data chunks
    course_data_chunks = process_course_data(timetable_data["courses"])
    for i, text in enumerate(course_data_chunks):
        split_text_chunks = text_splitter.split_text(text)
        all_text_chunks += [{"label": f"course_{i}_chunk_{j}", "content": chunk} for j, chunk in enumerate(split_text_chunks)]
    
    return all_text_chunks


#print(all_text_chunks)


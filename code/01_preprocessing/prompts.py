prompt = '''
### **Role Assignment**

You are an expert in analyzing user digital behavior data in online learning environments. User digital behavior has a hierarchical and complex structure. Your specialty is organizing the meanings of detailed behaviors and summarizing them into several simple keywords to understand the hierarchical structure of digital behavior.

### **Task Objective**
Analyze multiple Description strings and transform them according to the given rules to extract only the core distinguishing identifiers that differentiate each item.

*** Data Background ***
- Source: User behavior logs generated during online problem-solving processes
- event_type: Major classification of actions performed by users (e.g., menu, combobox)
- description: Specific parameters or targets of the corresponding behavior (e.g., index=10, id=inbox, value=Felix)
    - description is expressed in the structure of 'SV(system variable)'='DI(detailed information)'
    - Multiple 'SV'='DI' pairs can be expressed in a single description, separated by '|'
    - 'DI' consists of keywords separated by '_'
    

## **Input Data Structure** ##

The input data has the following JSON structure:
{
    "event_type1" : [
        "id=...",
        "id=...",
        ...
    ],
    "event_type2" : [
        "id=...",
        "index=...",
        ...
    ]
}

**Important**:
- Each event_type(e.g., 'combobox', 'toolbar') is an independent group
- Transformation is performed independently within each group only
- Do not compare or influence between different groups

## **Core Principles** ##

1. **Similar Pattern**
Descriptions are considered "similar pattern" when:
- The meaning of descriptions expressed by each keyword is the same
- The structure of keywords separated by delimiters is identical

2. **Delimiter Usage Rules**
    - '-' (hyphen): Connect keywords that constitute common keywords
    - '_' (underscore): common, different keywords를 연결하는 구분자

3. **Common Keywords and Different Keywords**
When comparing multiple Descriptions within the same group:
    - Common keywords: Keywords that overlap in Descriptions grouped by similar patterns
    - Different keywords: Keywords that do not overlap in Descriptions grouped by similar patterns


## ** Keyword Transformation Rules** ##
- All Descriptions with the same event_type group and similar pattern follow the rules below.

### Priority 1: Special Character Processing
    - '|' (pipe): Convert to '_' by default to connect
    - Extract detailed information from Description
    - If detailed variable is a single number, connect 'SI' + 'DI'
    
    id=pg10|action=open -> id=pg10_open (connect only detailed information 'open')
    id=inclue_btn30|index=4 -> id=include_btn30_index4 (connect system variable 'index' with single number 4)


### Priority 2: Classify Description into common keywords and different keywords.
    1. When keywords separated by '_' have the same value at the same position
    - toolbar_bkm_btn, toolbar_help_btn
    
    ID1 keywords : toolbar, bkm, btn
    ID2 keywords : toolbar, help, btn
    
    common keyword : toolbar, btn 
    different keyword : bkm, help
    
    - sort_index10, sort_index4
    
    ID1 keywords : sort, index10
    ID2 keywords : sort, index4
    
    common keyword : sort 
    different keyword : index10, index4 

    2. When there are overlapping values in simply connected values without delimiters
    - Find and separate common strings of 3 or more consecutive characters in Description
        
    - removecancel, movecancel
    
    ID1 keywords : removecancel
    ID2 keywords : movecancel

    common keyword : cancel 
    different keyword : remove, move
    
    - createfolder, createnewinput
    
    ID1 keywords : createfolder
    ID2 keywords : createnewinput

    common keyword : create 
    different keyword : folder, newinput
    
    3. When keywords containing '-' overlap
    - file-menu, ss-file-menu, wp-file-menu
    
    ID1 keywords : file-menu
    ID2 keywords : ss-file-menu
    ID3 keywords : wp-file-menu
    
    common keyword : file-menu
    different keyword : ss, wp

    (Additional Explanation)
    - 'file' and 'menu' are connected by '-' to form the keyword 'file-menu' which has an independent semantic unit.
    - Therefore, ss-file-menu and wp-file-menu can be said to have the common keyword 'file-menu'.


### Priority 3: Apply Keyword Processing Rules
    Basic Rule : Common keywords and Different keywords must be connected with '_'
        
    - Common keywords undergo different transformations depending on their position
    
    A. Common Keywords Located in the "Middle"
        - Connect the common keyword and the following keyword with '-'
        
        **Input:**
        pg4_pu6_okbtn
        pg7_pu6_cancelbtn
        
        **Analysis:**
        - common keyword(located in the middle) : pu6 
        - different keywords: okbtn, cancelbtn
        
        **Output:**
        pg4_pu6-okbtn
        pg7_pu6-cancelbtn

    B. Common Keywords Located at the "End"
        - Connect the immediately preceding keyword and the common keyword with '-'

        **Input:**
        inbox_bkm_btn
        outbox_help_btn
        
        **Analysis:**
        - common keyword(located at the end) : btn 
        - different keywords: inbox, outbox, bkm, help 
        
        **Output:**
        inbox_bkm-btn
        outbox_help-btn
        
    C. Common Keywords Located at the "Front"
        - Even if it's a common keyword, maintain it without connecting with '-'
        
        (Example 1)
        **Input:**
        mailbox_index3
        mailbox_index10
        
        **Analysis:**
        - common keyword(located at the front) : mailbox 
        - different keywords: index3, index10 
        
        **Output:**
        mailbox_index3
        mailbox_index10

        (Example 2)
        **Input:**
        mail_view_index30
        mail_view_index45
        
        **Analysis:**
        - common keywords(located at the front, consecutive) : mail, view
        - different keywords : index30, index45
        
        **Output:**
        mail-view_index30
        mail-view_index45

    D. Common Keywords Located in Multiple Places
        - Apply rules A, B, C sequentially
        
        (Example 1)
        **Input:**
        u07_pg4_pu6_okbtn
        u07_pg7_pu6_cancelbtn
        
        **Analysis:**
        - common keyword (located at the front, rule C) : u07
        - common keyword (located in the middle, rule A) : pu6
        - different keywords : pg4, pg7, okbtn, cancelbtn
        
        ** A.Output:**
        u07-pg4_pu6_okbtn
        u07-pg7_pu6_cancelbtn
        
        ** B.Output:**
        u07-pg4_pu6-okbtn
        u07-pg7_pu6-cancelbtn
        
        (Example 2)
        **Input:**
        toolbar_bkm_btn
        toolbar_help_btn
        
        **Analysis:**
        - common keyword (located at the front, rule C) : toolbar
        - common keyword (located at the end, rule B) : btn 
        - different keywords : bkm, help
        
        ** A.Output:**
        toolbar_bkm_btn
        toolbar_help_btn
        
        ** C.Output:**
        toolbar_bkm-btn
        toolbar_help-btn

        (Example 3)
        **Input:**
        case1_pg10_pu6
        case3_pg10_pu6
        
        **Analysis:**
        - common keyword (located in the middle, rule A) : pg10
        - common keyword (located at the end, rule B) : pu6
        - different keywords : case1, case3
        
        ** B, C.Output:**
        case1_pg10-pu6
        case3_pg10-pu6

    (Additional)
    Exception Case - This rule is the strongest rule in priority 3
        - If the description is a simply connected value without delimiters, exceptionally connect common keywords and different keywords with '_'(Step2-2 case)
        
        (Example 1)
        **Input:**
        removecancel
        movecancel

        **Analysis:**
        - common keyword : cancel
        - different keyword : remove, move
        
        **Output:**
        remove_cancel
        move_cancel
        
        (Example 2)
        **Input:**
        createfolder
        createnewinput
    
        **Analysis:**
        - common keyword : create 
        - different keyword : folder, newinput
        
        **Output:**
        create_folder
        create_newinput

    Exception Case - This rule is the strongest rule in priority 3
        - If keywords starting with 'u' ('u00', 'u01a', 'u07', ...), connect to the following keyword with '-'

        (Example)
        **Input:**
        u07_pg4_item19
        u07_pg7_item21
        u06a_popup1_txt4
        u06a_popup1_txt3
        u21p2pu5_txt1
        
        **Analysis:**
        - keywords starting with 'u' : u07, u06a, u21p2pu5
        - different keywords : pg4, pg7, item19, item21, txt4, txt3, txt1
        
        **Output:**
        u07-pg4_item19
        u07-pg7_item21
        u06a-popup1_txt4
        u06a-popup1_txt4
        u21p2pu5-txt1

## Important Notes ##

1. **Check the Additional Rules**
    - Common keywords consisting of multiple keywords must be grouped with '-'
        (ex)
        common keyword: 'inquiry_box' (keyword consisting of 'inquiry' and 'box')
        convert to 'inquiry-box'
    - 'Character+Number' are classified as different keywords even if the Character is the same
        (ex) different keywords: 'page10', 'page22', 'page30'
        
2. **Do Not Separate 'Character+Number' Combinations**
Semantic units containing numbers are maintained as one.
    - 'txt13' -> 'txt_13' (X)
    - 'popup1' -> 'popup_1' (X)
    - 'menu2' -> 'menu_2' (X)
    - 'pg4' -> 'pg_4' (X)

3. **Remove Duplicate Semantic Units**
    - When the same semantic unit repeats, keep only one
    - Example: 'add_bookmark_bookmark_title' -> 'add_bookmark_title'

4. **Maintain Independent Items**
    - Single Descriptions that don't form a group retain their original form
    - Example: 'searchfield', 'case-find'

## Application Method
For a given list of Descriptions:
1. Separate by event_type action group
2. Process independently within each group:
    a. Scan all Descriptions to identify patterns
    b. Group Descriptions with similar structures
    c. Step1. Handle '|' and 'key=value' formats 
    d. Step2. Identify common parts and different parts of each group
    e. Step3. Restructure in appropriate format according to given rules


*** Output Format ****
- Return results in markdown table format with columns : event_type, description, substitute
- Maintain original data values for event_type, description
- Return the restructured final value in substitute

** event_type | description | substitute **
textbox_onfocus | id=createfoldernameinput | create-folder_nameinput
textbox_onfocus | id=create_folder_name | create-folder_name
textbox_onfocus | id=createfolderwarningtext2 | create-folder_warningtext2
textbox_onfocus | u021_pg4_menu2 | u021-pg4_menu2
textbox_onfocus | u021_pg4_menu1 | u021-pg4_menu1
textbox_onfocus | u021_pg2_menu1 | u021-pg2_menu1
menu | id=edit-menu | edit-menu
menu | id=wb-edit-menu | wb_edit-menu
menu | id=ss-edit-menu | ss_edit-menu
combobox | id=mail_view_index45 | mail-view_index45
combobox | id=mail_view_index30 | mail-view_index30
combobox | id=inquiry_1_interaction_ddmenupopup|index=5 | inquiry-1-interaction-ddmenupopup_index5
toolbar | id=toolbar_forward_btn | toolbar_forward-btn
toolbar | id=toolbar_bkm_btn | toolbar_bkm-btn
button | id=popup1_pg3_pu7|action=open | popup1_pg3_pu7_open
button | id=popup4_pd5_pu10|index=10 | popup4_pd5_pu10_index10
textbox_onfocus | id=clientzone_box1_box2_textbox1 | clientzone-box1_box2-textbox1
textbox_onfocus | id=clientzone_box1_box1_textbox1 | clientzone-box1_box1-textbox1


**Do not output any text above or below the table. Output only the dataframe table.**
'''

verify_prompt = '''
**Role Assignment**

You are a data quality manager. You must evaluate from the perspective of a quality manager who verifies whether data preprocessing results follow the given rules well and identifies errors or missing parts.

**Task Objective and Evaluation Criteria**

Compare the input data preprocessing rules with the preprocessing results to verify whether preprocessing was performed correctly. Evaluate based on preprocessing rule compliance and completeness.

**Output Format**

- Add a new column called 'fixed' to rows that do not follow the data preprocessing rules and output
    - Do not output rows without issues, and if there are problems, enter the corrected version in the 'fixed' column
    
(Output Example)
**event_type | description | count | substitute | fixed**
menu | id=wp-file-menu | 1354 | wp-file-menu | wp_file-menu
menu | id=ss-file-menu | 9 | ss-file-menu | ss_file-menu
menu | id=file-menu | 1354 | file_menu | file-menu

**Quality Criteria**

- Does it faithfully follow the given data preprocessing rules? (e.g Step1, Step2, Step3)
- Does it comply well with the Important Notes?
- Does it output well according to the examples shown in the Output Example?

*** Data Background ***
- Source: User behavior logs generated during online problem-solving processes
- event_type: Major classification of actions performed by users (e.g., menu, combobox)
- description: Specific parameters or targets of the corresponding behavior (e.g., index=10, id=inbox, value=Felix)
    - description is expressed in the structure of 'SV(system variable)'='DI(detailed information)'
    - Multiple 'SV'='DI' pairs can be expressed in a single description, separated by '|'
    - 'DI' consists of keywords separated by '_'
    

## **Input Data Structure** ##

The input data has the following JSON structure:
{
    "event_type1" : [
        "id=...",
        "id=...",
        ...
    ],
    "event_type2" : [
        "id=...",
        "index=...",
        ...
    ]
}

**Important**:
- Each event_type(e.g., 'combobox', 'toolbar') is an independent group
- Transformation is performed independently within each group only
- Do not compare or influence between different groups

## **Core Principles** ##

1. **Similar Pattern**
Descriptions are considered "similar pattern" when:
- The meaning of descriptions expressed by each keyword is the same
- The structure of keywords separated by delimiters is identical

2. **Delimiter Usage Rules**
    - '-' (hyphen): Connect keywords that constitute common keywords
    - '_' (underscore): common, different keywords를 연결하는 구분자

3. **Common Keywords and Different Keywords**
When comparing multiple Descriptions within the same group:
    - Common keywords: Keywords that overlap in Descriptions grouped by similar patterns
    - Different keywords: Keywords that do not overlap in Descriptions grouped by similar patterns


## ** Keyword Transformation Rules** ##
- All Descriptions with the same event_type group and similar pattern follow the rules below.

### Priority 1: Special Character Processing
    - '|' (pipe): Convert to '_' by default to connect
    - Extract detailed information from Description
    - If detailed variable is a single number, connect 'SI' + 'DI'
    
    id=pg10|action=open -> id=pg10_open (connect only detailed information 'open')
    id=inclue_btn30|index=4 -> id=include_btn30_index4 (connect system variable 'index' with single number 4)


### Priority 2: Classify Description into common keywords and different keywords.
    1. When keywords separated by '_' have the same value at the same position
    - toolbar_bkm_btn, toolbar_help_btn
    
    ID1 keywords : toolbar, bkm, btn
    ID2 keywords : toolbar, help, btn
    
    common keyword : toolbar, btn 
    different keyword : bkm, help
    
    - sort_index10, sort_index4
    
    ID1 keywords : sort, index10
    ID2 keywords : sort, index4
    
    common keyword : sort 
    different keyword : index10, index4 

    2. When there are overlapping values in simply connected values without delimiters
    - Find and separate common strings of 3 or more consecutive characters in Description
        
    - removecancel, movecancel
    
    ID1 keywords : removecancel
    ID2 keywords : movecancel

    common keyword : cancel 
    different keyword : remove, move
    
    - createfolder, createnewinput
    
    ID1 keywords : createfolder
    ID2 keywords : createnewinput

    common keyword : create 
    different keyword : folder, newinput
    
    3. When keywords containing '-' overlap
    - file-menu, ss-file-menu, wp-file-menu
    
    ID1 keywords : file-menu
    ID2 keywords : ss-file-menu
    ID3 keywords : wp-file-menu
    
    common keyword : file-menu
    different keyword : ss, wp

    (Additional Explanation)
    - 'file' and 'menu' are connected by '-' to form the keyword 'file-menu' which has an independent semantic unit.
    - Therefore, ss-file-menu and wp-file-menu can be said to have the common keyword 'file-menu'.


### Priority 3: Apply Keyword Processing Rules
    Basic Rule : Common keywords and Different keywords must be connected with '_'
        
    - Common keywords undergo different transformations depending on their position
    
    A. Common Keywords Located in the "Middle"
        - Connect the common keyword and the following keyword with '-'
        
        **Input:**
        pg4_pu6_okbtn
        pg7_pu6_cancelbtn
        
        **Analysis:**
        - common keyword(located in the middle) : pu6 
        - different keywords: okbtn, cancelbtn
        
        **Output:**
        pg4_pu6-okbtn
        pg7_pu6-cancelbtn

    B. Common Keywords Located at the "End"
        - Connect the immediately preceding keyword and the common keyword with '-'

        **Input:**
        inbox_bkm_btn
        outbox_help_btn
        
        **Analysis:**
        - common keyword(located at the end) : btn 
        - different keywords: inbox, outbox, bkm, help 
        
        **Output:**
        inbox_bkm-btn
        outbox_help-btn
        
    C. Common Keywords Located at the "Front"
        - Even if it's a common keyword, maintain it without connecting with '-'
        
        (Example 1)
        **Input:**
        mailbox_index3
        mailbox_index10
        
        **Analysis:**
        - common keyword(located at the front) : mailbox 
        - different keywords: index3, index10 
        
        **Output:**
        mailbox_index3
        mailbox_index10

        (Example 2)
        **Input:**
        mail_view_index30
        mail_view_index45
        
        **Analysis:**
        - common keywords(located at the front, consecutive) : mail, view
        - different keywords : index30, index45
        
        **Output:**
        mail-view_index30
        mail-view_index45

    D. Common Keywords Located in Multiple Places
        - Apply rules A, B, C sequentially
        
        (Example 1)
        **Input:**
        u07_pg4_pu6_okbtn
        u07_pg7_pu6_cancelbtn
        
        **Analysis:**
        - common keyword (located at the front, rule C) : u07
        - common keyword (located in the middle, rule A) : pu6
        - different keywords : pg4, pg7, okbtn, cancelbtn
        
        ** A.Output:**
        u07-pg4_pu6_okbtn
        u07-pg7_pu6_cancelbtn
        
        ** B.Output:**
        u07-pg4_pu6-okbtn
        u07-pg7_pu6-cancelbtn
        
        (Example 2)
        **Input:**
        toolbar_bkm_btn
        toolbar_help_btn
        
        **Analysis:**
        - common keyword (located at the front, rule C) : toolbar
        - common keyword (located at the end, rule B) : btn 
        - different keywords : bkm, help
        
        ** A.Output:**
        toolbar_bkm_btn
        toolbar_help_btn
        
        ** C.Output:**
        toolbar_bkm-btn
        toolbar_help-btn

        (Example 3)
        **Input:**
        case1_pg10_pu6
        case3_pg10_pu6
        
        **Analysis:**
        - common keyword (located in the middle, rule A) : pg10
        - common keyword (located at the end, rule B) : pu6
        - different keywords : case1, case3
        
        ** B, C.Output:**
        case1_pg10-pu6
        case3_pg10-pu6

    (Additional)
    Exception Case - This rule is the strongest rule in priority 3
        - If the description is a simply connected value without delimiters, exceptionally connect common keywords and different keywords with '_'(Step2-2 case)
        
        (Example 1)
        **Input:**
        removecancel
        movecancel

        **Analysis:**
        - common keyword : cancel
        - different keyword : remove, move
        
        **Output:**
        remove_cancel
        move_cancel
        
        (Example 2)
        **Input:**
        createfolder
        createnewinput
    
        **Analysis:**
        - common keyword : create 
        - different keyword : folder, newinput
        
        **Output:**
        create_folder
        create_newinput

    Exception Case - This rule is the strongest rule in priority 3
        - If keywords starting with 'u' ('u00', 'u01a', 'u07', ...), connect to the following keyword with '-'

        (Example)
        **Input:**
        u07_pg4_item19
        u07_pg7_item21
        u06a_popup1_txt4
        u06a_popup1_txt3
        u21p2pu5_txt1
        
        **Analysis:**
        - keywords starting with 'u' : u07, u06a, u21p2pu5
        - different keywords : pg4, pg7, item19, item21, txt4, txt3, txt1
        
        **Output:**
        u07-pg4_item19
        u07-pg7_item21
        u06a-popup1_txt4
        u06a-popup1_txt4
        u21p2pu5-txt1

## Important Notes ##

1. **Check the Additional Rules**
    - Common keywords consisting of multiple keywords must be grouped with '-'
        (ex)
        common keyword: 'inquiry_box' (keyword consisting of 'inquiry' and 'box')
        convert to 'inquiry-box'
    - 'Character+Number' are classified as different keywords even if the Character is the same
        (ex) different keywords: 'page10', 'page22', 'page30'
        
2. **Do Not Separate 'Character+Number' Combinations**
Semantic units containing numbers are maintained as one.
    - 'txt13' -> 'txt_13' (X)
    - 'popup1' -> 'popup_1' (X)
    - 'menu2' -> 'menu_2' (X)
    - 'pg4' -> 'pg_4' (X)

3. **Remove Duplicate Semantic Units**
    - When the same semantic unit repeats, keep only one
    - Example: 'add_bookmark_bookmark_title' -> 'add_bookmark_title'

4. **Maintain Independent Items**
    - Single Descriptions that don't form a group retain their original form
    - Example: 'searchfield', 'case-find'

## Application Method
For a given list of Descriptions:
1. Separate by event_type action group
2. Process independently within each group:
    a. Scan all Descriptions to identify patterns
    b. Group Descriptions with similar structures
    c. Step1. Handle '|' and 'key=value' formats 
    e. Step2. Identify common parts and different parts of each group
    f. Step3. Restructure in appropriate format according to given rules


*** Output Format ****
- Return results in markdown table format with columns : event_type, description, substitute
- Maintain original data values for event_type, description
- Return the restructured final value in substitute

** event_type | description | substitute **
textbox_onfocus | id=createfoldernameinput | create-folder_nameinput
textbox_onfocus | id=create_folder_name | create-folder_name
textbox_onfocus | id=createfolderwarningtext2 | create-folder_warningtext2
textbox_onfocus | u021_pg4_menu2 | u021-pg4_menu2
textbox_onfocus | u021_pg4_menu1 | u021-pg4_menu1
textbox_onfocus | u021_pg2_menu1 | u021-pg2_menu1
menu | id=edit-menu | edit-menu
menu | id=wb-edit-menu | wb_edit-menu
menu | id=ss-edit-menu | ss_edit-menu
combobox | id=mail_view_index45 | mail-view_index45
combobox | id=mail_view_index30 | mail-view_index30
combobox | id=inquiry_1_interaction_ddmenupopup|index=5 | inquiry-1-interaction-ddmenupopup_index5
toolbar | id=toolbar_forward_btn | toolbar_forward-btn
toolbar | id=toolbar_bkm_btn | toolbar_bkm-btn
button | id=popup1_pg3_pu7|action=open | popup1_pg3_pu7_open
button | id=popup4_pd5_pu10|index=10 | popup4_pd5_pu10_index10
textbox_onfocus | id=clientzone_box1_box2_textbox1 | clientzone-box1_box2-textbox1
textbox_onfocus | id=clientzone_box1_box1_textbox1 | clientzone-box1_box1-textbox1


**Do not output any text above or below the table. Output only the dataframe table.**
'''
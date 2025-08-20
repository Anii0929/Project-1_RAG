Refactor @backend/ai_generator.py to support sequential tool calling where claude can make up to 2 tool calls in separate API rounds.

Current Behavior:
- Claude makes 1 tool call -> tools are removed form API params -> final response
- If Claude wants another tool call after seeing results it can't (gets empty response)

Desired Behavior:
- Each tool call should be a separate API request where Claude can reason about previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part questions, or when information different courses/lessons is needed

Example flow:
1. User: "Search for a course that discusses the same topic as lesson 4 of course X"
2. Claude: get course outlien for courseX -> gerts title of leson 4
3. Caluade: uses the title to search for a course that discusses the same topic -> returns course information
4. Claude: provides complete answer

Requirements:
- Maximum 2 squential roudns per user query
- Terminate when: (a) 2 roudns completed (b) Claudes response has no tool_use blocks, or (c) tool call fails
- Preserve conversation context between rounds
- Handle tool execution errors gracefully

Notes:
- update the system prompt in @backend/ai_generator.py
- update the test @backend/tests/test_ai_generator.py
- Write tests that verify the external behavior (API calls made, tools executed, results returned) rather than internal state details

Use two parallel subagents to brainstorm possible plans. Do no implement any code. 
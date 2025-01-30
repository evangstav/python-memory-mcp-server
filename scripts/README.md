# Memory Migration Script

This script migrates your existing memory.jsonl file to comply with the new validation rules.

## What it does

1. Reads the existing JSONL format where each line is either:
   ```json
   {"type": "entity", "name": "example", "entityType": "person", "observations": ["obs1"]}
   ```
   or
   ```json
   {"type": "relation", "from": "entity1", "to": "entity2", "relationType": "knows"}
   ```

2. Converts entity and relation names to the new format:
   - Lowercase with hyphens
   - No spaces or special characters
   - Must start with a letter
   - Example: "John Doe" -> "john-doe"

3. Normalizes entity types to valid categories:
   - person
   - concept
   - project
   - document
   - tool
   - organization
   - location
   - event

4. Normalizes relation types to valid verbs:
   - knows
   - contains
   - uses
   - created
   - belongs-to
   - depends-on
   - related-to

5. Validates and deduplicates observations

## Common Type Mappings

### Entity Types
- individual, user, human -> person
- doc, documentation -> document
- app, application, software -> tool
- group, team, company -> organization
- place, area -> location
- meeting, appointment -> event
- residence, property -> location
- software_project -> project
- dataset -> document
- health_record -> document
- meal -> document
- travel_event -> event
- pet -> concept
- venue -> location

### Relation Types
- knows_about -> knows
- contains_item, has -> contains
- uses_tool -> uses
- created_by, authored -> created
- belongs_to_group, member_of -> belongs-to
- depends_upon, requires -> depends-on
- related -> related-to
- works_at -> belongs-to
- owns -> created
- friend -> knows

## Usage

1. Make sure your memory.jsonl file is in the project root directory

2. Run the migration script:
   ```bash
   ./scripts/migrate_memory.py
   ```

3. The script will:
   - Read memory.jsonl line by line
   - Convert all data to the new format
   - Validate the migrated data
   - Write the result to memory.jsonl.new
   - Report any errors or issues

4. Review the output file and error messages

5. If satisfied with the migration, replace your old memory file:
   ```bash
   mv memory.jsonl.new memory.jsonl
   ```

## Error Handling

The script will:
- Report any entities or relations that couldn't be migrated
- Continue processing even if some items fail
- Validate the entire graph before saving
- Preserve your original file by writing to .new file
- Track name changes to ensure relations are updated correctly

## Example Output

```
Migrating memory.jsonl to memory.jsonl.new...

Migration complete:
- Successfully migrated 42 entities
- Encountered 2 errors

Errors encountered:
- Error migrating line: {"type": "entity", "name": "Invalid!Name"...}
  Error: Invalid entity name format
- Error migrating line: {"type": "relation", "from": "A"...}
  Error: Invalid relation type

Migrated data written to memory.jsonl.new
Please verify the output before replacing your original memory file.
```

## Validation Rules

### Entity Names
- Must start with a lowercase letter
- Can contain lowercase letters, numbers, and hyphens
- Maximum length of 100 characters
- Must be unique within the graph

### Observations
- Non-empty strings
- Maximum length of 500 characters
- Must be unique per entity
- Factual and objective statements

### Relations
- Both source and target entities must exist
- Self-referential relations not allowed
- No circular dependencies
- Must use predefined relation types

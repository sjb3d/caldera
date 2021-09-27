uint coherent_hash(uint key, uint age)
{
    uint index = key + HASH_TABLE_INFO.offsets[age - 1];
    uint count = HASH_TABLE_INFO.entry_count;
    return index % count;
}

#ifdef HASH_TABLE_ENTRIES_WRITE
bool insert_entry(uint key, uint data)
{
    Entry entry = make_entry(1, key, data);
    for (;;) {
        uint entry_index = coherent_hash(get_key(entry), get_age(entry));
        Entry prev = make_entry(atomicMax(HASH_TABLE_ENTRIES_WRITE[entry_index], entry.bits));
        if (entry.bits > prev.bits) {
            // we displaced the (younger) entry here, so update max age for the new entry
            if (HASH_TABLE_INFO.store_max_age != 0) {
                uint max_age_index = coherent_hash(get_key(entry), 1);
                atomicMax(HASH_TABLE_MAX_AGES[max_age_index], get_age(entry));
            }

            // find a new home for the entry that was displaced (if non-empty)
            entry = prev;
            if (get_age(entry) == 0) {
                // insert succeeded
                return true;
            }
        }
        if (get_age(entry) == MAX_AGE) {
            // insert failed
            return false;
        }
        entry = increment_age(entry);       
    }
}
#endif

#ifdef HASH_TABLE_ENTRIES_READ
bool get_entry(uint key, out uint data)
{
    uint max_age = MAX_AGE;
    if (HASH_TABLE_INFO.store_max_age != 0) {
        uint max_age_index = coherent_hash(key, 1);
        max_age = HASH_TABLE_MAX_AGES[max_age_index];
    }
    for (uint age = 1; age <= max_age; ++age) {
        uint entry_index = coherent_hash(key, age);
        Entry entry = make_entry(HASH_TABLE_ENTRIES_READ[entry_index]);
        if (get_age(entry) == 0) {
            // entry is empty, no need to check older ones
            break;
        }
        if (get_key(entry) == key) {
            data = get_data(entry);
            return true;
        }
    }
    return false;
}
#endif

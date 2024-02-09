package com.example.thesisocr

/**
 * TextHandler class for processing extracted text
 */

/**
 * Format of National Identity Card:
 * 1. Last Name (String)
 * 2. Given Name (String)
 * 3. Middle Name (String)
 * 3. Identity Number (xxxx-xxxx-xxxx) (String)
 * 4. Date of Birth (MMMM-DD-YYYY) (String)
 * 5. Address (String)
 */

/**
 * Methods for determining the type of string extracted from the image.
 * Identity Number and Date of Birth uses pattern-matching.
 * Explore use of named entity recognition (NER) for extracting names information.
 * Explore use of regular expressions for extracting address information.
 */
class TextHandler {
    fun determineStringType(text: String): String {
        return when {
            isIdentityNumber(text) -> "Identity Number"
            isDateOfBirth(text) -> "Date of Birth"
            else -> "Unknown"
        }
    }
    private fun isIdentityNumber(text: String): Boolean {
        val regex = Regex("^[0-9]{4}-[0-9]{4}-[0-9]{4}\$")
        return regex.matches(text)
    }
    private fun isDateOfBirth(text: String): Boolean {
        val regex = Regex("^[A-Z][a-z]{3,8}-[0-9]{1,2}-[0-9]{4}\$")
        return regex.matches(text)
    }
}
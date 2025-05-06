import type { Meta, StoryObj } from '@storybook/react';
import Section from './Section';
import { Box, Typography } from '@mui/material';

type Story = StoryObj<typeof Section>;

const meta: Meta<typeof Section> = {
  title: 'Components/Section',
  component: Section,
  args: {
    title: 'Basic Section',
    subtitle: 'Subtitle',
    body: 'This is a basic section with title, subtitle and body text.',
  },
};

export default meta;

export const Default: Story = {};

export const WithoutSubtitle: Story = {
  args: {
    title: 'Section Without Subtitle',
    body: 'This is a section with title and body, but no subtitle.',
  },
};

export const WithChildren: Story = {
  args: {
    title: 'Section With Children',
    subtitle: 'With additional content',
    body: 'This section contains child components.',
    children: (
      <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
        <Typography variant="body2">This is a child component inside the section.</Typography>
      </Box>
    ),
  },
};

export const LongContent: Story = {
  args: {
    title: 'Section With Long Content',
    subtitle: 'Expanded example',
    body: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisi vel consectetur interdum, nisl nisi consectetur nisi, eget consectetur nisi nisi vel nisi. Nullam euismod, nisi vel consectetur interdum, nisl nisi consectetur nisi, eget consectetur nisi nisi vel nisi.',
  },
};

export const WithoutBody: Story = {
  args: {
    title: 'Section Without Body',
    subtitle: 'Only header components',
    children: (
      <Typography variant="body2" color="text.secondary">
        This section has no body text, only children.
      </Typography>
    ),
  },
};
